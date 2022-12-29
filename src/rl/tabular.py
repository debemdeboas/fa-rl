from typing import Any, Collection, List, NoReturn, Sequence, overload
from random import Random
from abc import abstractmethod
from math import log2, exp
from gym.core import Env
from tqdm import tqdm
import numpy as np

from rl.agent import RLAgent, State, Action, softmax


class TabularQLearner(RLAgent):
    """
    A simple Tabular Q-Learning agent.
    """
    def __init__(self,
                 env: Env,
                 episodes: int = 500,
                 decaying_eps: bool = True,
                 eps: float = 1.0,
                 alpha: float = 0.5,
                 decay: float = 0.000002,
                 gamma: float = 0.9,
                 rand: Random = Random(),
                 **kwargs):
        super().__init__(env, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha, decay=decay, gamma=gamma, rand=rand)
        # print(f"Env Shape {env.action_space.shape}, {env.action_space.n}")
        self.actions = env.action_space.n

        self.q_table = {}

        # hyperparameters
        self.episodes = episodes
        self.gamma = gamma
        self.decay = decay
        self.c_eps = eps
        self.base_eps = eps
        if decaying_eps:
            def epsilon():
                self.c_eps = max((self.episodes - self.step)/self.episodes, 0.01)
                return self.c_eps
            self.eps = epsilon
        else:
            self.eps = lambda: eps
        self.decaying_eps = decaying_eps
        self.alpha = alpha
        self.last_state = None
        self.last_action = None

    def states_in_q(self) -> List:
        """Returns the states stored in the q_values table

        Returns:
            List: The states for which we have a mapping in the q-table
        """
        return self.q_table.keys()

    def policy(self, state: State) -> Action:
        """Returns the greedy deterministic policy for the specified state

        Args:
            state (State): the state for which we want the action

        Raises:
            InvalidAction: Not sure about this one

        Returns:
            Any: The greedy action learned for state
        """
        return self.best_action(state)

    def epsilon_greedy_policy(self, state: State) -> Action:
        """Returns the epsilon-greedy policy

        Args:
            state (State): The state for which to return the epsilon greedy policy

        Returns:
            Any: The action to be taken
        """
        eps = self.eps()
        if self._random.random() <= eps:
            action = self._random.randint(0, self.actions-1)
        else:
            action = self.policy(state)
        return action

    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        if state not in self.q_table:
            self.add_new_state(state)
            # If we query a state we have not visited, return a uniform distribution
            # return softmax([0]*self.actions)
        return softmax(self.q_table[state])

    def add_new_state(self, state: State):
        # self.q_table[state] = [1. for _ in range(self.actions)]
        self.q_table[state] = [0.]*self.actions

    def get_all_q_values(self, state: State) -> List[float]:
        if state in self.q_table:
            return self.q_table[state]
        else:
            return [0.]*self.actions

    def best_actions(self, state: State) -> Sequence[int]:
        """Returns a list with the best actions for a particular state

        Args:
            state (State): The state for which we need the best actions

        Returns:
            Sequence[int]: a list with the best actions
        """
        if state not in self.q_table:
            self.add_new_state(state)
        q_next = np.array(self.q_table[state])
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()
        return best_actions

    def best_action(self, state: State) -> Action:
        """Returns the best action for a state, breaking ties randomly

        Args:
            state (State): The state in which to extract the best action

        Returns:
            int: The index of the best action
        """
        if state not in self.q_table:
            self.add_new_state(state)
            # self.q_table[state] = [0 for _ in range(self.actions)]
        # return np.argmax(self.q_table[state])
        best_actions = self.best_actions(state)

        return self._random.choice(best_actions)

    def get_max_q(self, state: State) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return np.max(self.q_table[state])

    def set_q_value(self, state: State, action: Any, q_value: float):
        if state not in self.q_table:
            self.add_new_state(state)
        self.q_table[state][action] = q_value

    def get_q_value(self, state: State, action: Any) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return self.q_table[state][action]

    def agent_start(self, state: State) -> Action:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        self.last_state = state
        self.last_action = self.epsilon_greedy_policy(state)
        return self.last_action

    def agent_step(self, reward: float, state: State) -> Action:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma*max_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        action = self.epsilon_greedy_policy(state)
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> NoReturn:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = - old_q

        new_q = old_q + self.alpha * (reward + td_error)
        self.set_q_value(self.last_state, self.last_action, new_q)


class TabularSarsaLearner(TabularQLearner):
    def agent_step(self, reward: float, state: State) -> Action:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        action = self.epsilon_greedy_policy(state)
        next_q = self.get_q_value(state, action)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma*next_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        self.last_state = state
        self.last_action = action
        return action


class TabularExpectedSarsaLearner(TabularQLearner):
    def agent_step(self, reward: float, state: State) -> Action:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        next_q = 0.0
        action = self.epsilon_greedy_policy(state)
        best_actions = self.best_actions(state)
        for a in range(self.actions):
            eps = self.eps()
            if a in best_actions:
                next_q += ((1.0 - eps) / len(best_actions) + eps / self.actions) * self.get_q_value(state, a)
            else:
                next_q += (eps / self.actions) * self.get_q_value(state, a)

        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma*next_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        self.last_state = state
        self.last_action = action
        return action


class TabularDynaQLearner(TabularQLearner):
    def __init__(self,
                 env: Env,
                 episodes: int = 500,
                 decaying_eps: bool = True,
                 eps: float = 1.0,
                 alpha: float = 0.5,
                 decay: float = 0.000002,
                 gamma: float = 0.9,
                 rand: Random = Random(),
                 planning_steps: int = 10,
                 **kwargs):
        self.planning_steps = planning_steps
        self.model = {}  # model is a dictionary of dictionaries, which maps states to actions to (reward, next_state) tuples

        super().__init__(env, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha, decay=decay, gamma=gamma, rand=rand, **kwargs)

    def update_model(self, past_state: State, past_action, state: State, reward: float):
        if past_state not in self.model:
            self.model[past_state] = {}
        self.model[past_state][past_action] = (state, reward)

    def planning_step(self):
        for i in range(self.planning_steps):
            past_state = self._random.choice(list(self.model.keys()))
            past_action = self._random.choice(list(self.model[past_state].keys()))
            state, reward = self.model[past_state][past_action]
            if state is None:
                td_error = - self.get_q_value(past_state, past_action)
            else:
                td_error = self.gamma*self.get_max_q(state) - self.get_q_value(past_state, past_action)
            new_q = self.get_q_value(past_state, past_action) + self.alpha*(reward + td_error)
            self.set_q_value(past_state, past_action, new_q)

    def agent_start(self, state: State) -> Action:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        self.last_state = state
        self.last_action = self.policy(state)
        return self.last_action

    def agent_step(self, reward: float, state: State) -> Action:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma*max_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        action = self.epsilon_greedy_policy(state)

        self.update_model(self.last_state, self.last_action, state, reward)
        self.planning_step()

        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> NoReturn:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = - old_q

        new_q = old_q + self.alpha * (reward + td_error)
        self.set_q_value(self.last_state, self.last_action, new_q)
        self.update_model(self.last_state, self.last_action, None, reward)