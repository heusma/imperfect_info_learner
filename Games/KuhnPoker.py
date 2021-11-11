import copy
import random
from enum import Enum
from typing import Tuple, List

from Game import Game, StateNode, State, DiscreteDistribution, ChanceNode, InfoSet, ActionSchema, Node, Leaf


class Cards(Enum):
    J = 0
    Q = 1
    K = 2


class KuhnPokerState(State):
    def __init__(self, turn: int, current_player: int, p0_card: Cards, p1_card: Cards, p0_bet: int, p1_bet: int):
        super().__init__(current_player)

        self.turn = turn
        self.current_player_folded = False

        self.p0_card = p0_card
        self.p1_card = p1_card
        self.p0_bet = p0_bet
        self.p1_bet = p1_bet

    def to_info_set(self) -> InfoSet:
        return KuhnPokerInfoSet(
            turn=self.turn,
            current_player=self.current_player,
            current_player_card=self.p0_card if self.current_player == 0 else self.p1_card,
            p0_bet=self.p0_bet,
            p1_bet=self.p1_bet,
        )


class BetAction:
    def __init__(self, bet: int):
        self.bet = bet

    def apply(self, state: KuhnPokerState):
        state = copy.deepcopy(state)
        if state.current_player == 0:
            state.p0_bet += self.bet
        else:
            state.p1_bet += self.bet

        state.turn += 1
        state.current_player = 1 if state.current_player == 0 else 0
        return state


class FoldAction:
    def apply(self, state: KuhnPokerState):
        state = copy.deepcopy(state)
        state.current_player_folded = True
        return state


actions = [FoldAction(), BetAction(0), BetAction(1)]


class KuhnPokerActionSchema(ActionSchema):
    def __init__(self, chance_nodes: List[ChanceNode], root_node: int = 0):
        super().__init__(chance_nodes, root_node)

    def sample(self) -> Tuple[List[float], List[float], ChanceNode, Node]:
        p, v, nn = self.root_node().sample()
        return [p], [v], self.root_node(), nn


class KuhnPokerInfoSet(InfoSet):
    def __init__(self, turn: int, current_player: int, current_player_card: Cards, p0_bet: int, p1_bet: int):
        self.turn = turn
        self.current_player = current_player
        self.current_player_folded = False

        self.current_player_card = current_player_card
        self.p0_bet = p0_bet
        self.p1_bet = p1_bet

    def get_action_schema(self) -> ActionSchema:
        return KuhnPokerActionSchema(
            [ChanceNode(DiscreteDistribution(3))],
        )


class KuhnPokerStateNode(StateNode):
    def __init__(self, state: KuhnPokerState):
        super().__init__(state)

    def act(self, action: List[int or float]) -> Tuple[List[float], StateNode or Leaf]:
        assert len(action) == 1
        # apply the action to the state
        actions = [FoldAction(), BetAction(0), BetAction(1)]
        picked_action = actions[action[0]]

        # deal with illegal moves.
        legal_moves = [actions[0]]
        if self.state.p0_bet == self.state.p1_bet:
            legal_moves.append(actions[1])
        if (self.state.p0_bet == 1 and self.state.p1_bet == 1) or (self.state.p0_bet == 1 and self.state.p1_bet == 2) or (self.state.p0_bet == 2 and self.state.p1_bet == 1):
            legal_moves.append(actions[2])
        if picked_action not in legal_moves:
            picked_action = random.choice(legal_moves)
        new_state = picked_action.apply(self.state)

        # is the game over?
        direkt_reward = [0.0, 0.0]
        winner = -1
        if new_state.current_player_folded is True:
            if new_state.current_player == 0:
                winner = 1
            if new_state.current_player == 1:
                winner = 0
        if (new_state.p1_bet == 2 and new_state.p0_bet == 2) or (
                new_state.p1_bet == 1 and new_state.p0_bet == 1 and new_state.turn >= 2):
            if new_state.p0_card.value > new_state.p1_card.value:
                winner = 0
            else:
                winner = 1
        if winner >= 0:
            if winner == 1:
                direkt_reward[0] = -new_state.p0_bet
                direkt_reward[1] = new_state.p0_bet
            else:
                direkt_reward[1] = -new_state.p1_bet
                direkt_reward[0] = new_state.p1_bet
            return direkt_reward, Leaf()
        else:
            return direkt_reward, KuhnPokerStateNode(new_state)


class KuhnPoker(Game):
    @staticmethod
    def start(options: dict) -> StateNode:
        p0_card, p1_card = random.sample(list(Cards), k=2)
        root_state = KuhnPokerState(0, 0, p0_card, p1_card, 1, 1)
        return KuhnPokerStateNode(root_state)
