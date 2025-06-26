import argparse
import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn

# Board representation utilities
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a chess.Board to 8x8x20 tensor."""
    planes = np.zeros((20, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = PIECE_TO_CHANNEL[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            row = square // 8
            col = square % 8
            planes[channel, row, col] = 1
    # castling rights
    planes[12].fill(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    planes[13].fill(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    planes[14].fill(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    planes[15].fill(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    # en-passant square
    if board.ep_square:
        row = board.ep_square // 8
        col = board.ep_square % 8
        planes[16, row, col] = 1
    # side to move
    planes[17].fill(1 if board.turn == chess.WHITE else -1)
    # half-move clock
    planes[18].fill(board.halfmove_clock / 100.0)
    # full-move number
    planes[19].fill(board.fullmove_number / 100.0)
    return torch.from_numpy(planes)


# Neural network modules
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNetPolicyValue(nn.Module):
    def __init__(self, blocks: int = 10, channels: int = 256, move_count: int = 4672):
        super().__init__()
        self.initial = nn.Conv2d(20, channels, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, move_count),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value.squeeze(1)


class DuelingDQN(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 5, move_count: int = 4672):
        super().__init__()
        self.initial = nn.Conv2d(20, channels, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.adv_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, move_count),
        )
        self.val_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblocks(x)
        adv = self.adv_head(x)
        val = self.val_head(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


# MCTS utilities
class MCTSNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count


def move_to_index(move: chess.Move) -> int:
    return hash(move) % 4672


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    for move in board.legal_moves:
        if move_to_index(move) == index:
            return move
    return random.choice(list(board.legal_moves))


def softmax_sample(policy):
    policy = np.array(policy, dtype=np.float64)
    policy = np.exp(policy - np.max(policy))
    policy /= np.sum(policy)
    return np.random.choice(len(policy), p=policy)


class MCTS:
    def __init__(self, network: ResNetPolicyValue, c_puct: float = 4.0, simulations: int = 800):
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations

    def search(self, board: chess.Board):
        root = MCTSNode(None, 0)
        self.expand(root, board)
        self.add_dirichlet_noise(root)

        for _ in range(self.simulations):
            node = root
            scratch_board = board.copy()
            search_path = [node]
            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                scratch_board.push(action)
                search_path.append(node)
            # Evaluation
            value = self.evaluate(node, scratch_board)
            # Backprop
            for n in reversed(search_path):
                n.value_sum += value
                n.visit_count += 1
                value = -value
        return root

    def select_child(self, node):
        best_score = -float("inf")
        best_action = None
        best_child = None
        total_visits = math.sqrt(node.visit_count)
        for action, child in node.children.items():
            q = child.value()
            u = self.c_puct * child.prior * total_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, node, board):
        state = board_to_tensor(board).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.network(state)
        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        for move in board.legal_moves:
            node.children[move] = MCTSNode(node, policy[move_to_index(move)])
        return value.item()

    def evaluate(self, node, board):
        if board.is_game_over():
            outcome = board.outcome()
            if outcome.winner is None:
                return 0
            return 1 if outcome.winner == board.turn else -1
        return self.expand(node, board)

    def add_dirichlet_noise(self, node, epsilon=0.25, alpha=0.3):
        moves = list(node.children.keys())
        if not moves:
            return
        noise = np.random.dirichlet([alpha] * len(moves))
        for n, move in zip(noise, moves):
            child = node.children[move]
            child.prior = child.prior * (1 - epsilon) + n * epsilon

    def get_policy(self, node, temperature=1.0):
        visits = np.array([child.visit_count for child in node.children.values()], dtype=np.float64)
        if temperature == 0:
            best = np.argmax(visits)
            policy = np.zeros_like(visits)
            policy[best] = 1.0
        else:
            visits = visits ** (1 / temperature)
            policy = visits / np.sum(visits)
        moves = list(node.children.keys())
        policy_map = np.zeros(4672, dtype=np.float32)
        for move, p in zip(moves, policy):
            policy_map[move_to_index(move)] = p
        best_move = moves[np.argmax(policy)]
        return policy_map, best_move


class SelfPlayAgent:
    def __init__(self, network, device="cpu"):
        self.network = network.to(device)
        self.device = device

    def play_game(self):
        board = chess.Board()
        positions, policies, rewards = [], [], []
        mcts = MCTS(self.network)
        while not board.is_game_over():
            root = mcts.search(board)
            policy, action = mcts.get_policy(root)
            positions.append(board_to_tensor(board))
            policies.append(policy)
            board.push(action)
        result = board.result()
        reward = 0
        if result == "1-0":
            reward = 1
        elif result == "0-1":
            reward = -1
        rewards = [reward for _ in positions]
        return positions, policies, rewards


Transition = namedtuple("Transition", "state action reward next_state done")


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = 0.6

    def add(self, transition, error):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max(max_prio, abs(error))
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err)


class DQNAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.policy_net = DuelingDQN().to(device)
        self.target_net = DuelingDQN().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay = PrioritizedReplayBuffer()
        self.steps = 0

    def select_action(self, board, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(list(board.legal_moves))
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        q_values = q_values[0].cpu().numpy()
        legal_moves = list(board.legal_moves)
        values = [q_values[move_to_index(m)] for m in legal_moves]
        return legal_moves[int(np.argmax(values))]

    def optimize(self, batch_size=64, gamma=0.99):
        if len(self.replay.buffer) < batch_size:
            return
        transitions, indices, weights = self.replay.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor([move_to_index(a) for a in batch.action]).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        if any(non_final_mask):
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(self.device)
        else:
            non_final_next_states = torch.empty(0, 20, 8, 8).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_q = torch.zeros(batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_q[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        target = reward_batch + gamma * next_q * (~torch.tensor(batch.done, dtype=torch.bool).to(self.device))
        loss = (q_values - target).pow(2) * weights.to(self.device)
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay.update_priorities(indices, prios.data.cpu().numpy())
        self.steps += 1
        if self.steps % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class UCIEngine:
    def __init__(self, agent):
        self.agent = agent
        self.board = chess.Board()

    def loop(self):
        print("id name DualChessAI")
        print("uciok")
        while True:
            try:
                cmd = input()
            except EOFError:
                break
            if cmd == "quit":
                break
            elif cmd.startswith("position"):
                self.handle_position(cmd)
            elif cmd.startswith("go"):
                move = self.agent.select_action(self.board, epsilon=0)
                print(f"bestmove {move}")

    def handle_position(self, cmd):
        if cmd.startswith("position startpos"):
            self.board = chess.Board()
            moves = cmd.split("moves")
            if len(moves) > 1:
                for mv in moves[1].strip().split():
                    self.board.push_uci(mv)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_value_net = ResNetPolicyValue().to(device)
    sp_agent = SelfPlayAgent(policy_value_net, device)
    q_agent = DQNAgent(device)

    for epoch in range(args.epochs):
        positions, policies, rewards = sp_agent.play_game()
        for i in range(len(positions)):
            state = positions[i]
            next_state = positions[i + 1] if i + 1 < len(positions) else None
            action = index_to_move(int(np.argmax(policies[i])), chess.Board())  # placeholder mapping
            reward = rewards[i]
            done = next_state is None
            transition = Transition(state, action, reward, next_state, done)
            q_agent.replay.add(transition, 1.0)
        q_agent.optimize()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        engine = UCIEngine(DQNAgent())
        engine.loop()
