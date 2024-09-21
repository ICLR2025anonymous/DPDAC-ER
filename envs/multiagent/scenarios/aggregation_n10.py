"""
Scenario: A position aggregation task
Note: In the individual observation, the relative position lists are sorted.
"""
import numpy as np
from learning.envs.multiagent.core_vec import World, Agent, Landmark
from learning.envs.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 10  # even number
        num_landmarks = 1
        # world.collaborative = True
        self.world_radius = 1.6  # camera range
        self.num_agents = num_agents
        self.n_others = 3
        self.np_rnd = np.random.RandomState(0)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.idx = i
            agent.collide = False
            agent.silent = True
            agent.size = 0.04
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        world.landmarks[0].size = 0.7
        # make initial conditions
        self.collision_threshold = world.agents[0].size + world.landmarks[0].size
        self.collision_threshold_agent = 2*world.agents[0].size
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        radius_all = 1.5 + np.random.uniform(-0.1, 0.1, self.num_agents)
        angle_all = np.random.uniform(-np.pi, np.pi, self.num_agents)

        agent_pos = np.zeros((self.num_agents, world.dim_p))
        for i in range(self.num_agents):
            theta_i = angle_all[i]
            p_x = radius_all[i]*np.cos(theta_i)
            p_y = radius_all[i]*np.sin(theta_i)
            agent_pos[i] = np.array((p_x, p_y))

        np.random.shuffle(agent_pos)

        for i in range(self.num_agents):
            world.agents[i].state.p_pos = agent_pos[i]
            world.agents[i].state.p_vel = np.zeros(world.dim_p)
            world.agents[i].state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = np.zeros(world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """
        Vectorized reward function
        """
        reward_vec = 0
        if agent == world.agents[0]:
            # reward 1 (target)
            a_pos = np.array([a.state.p_pos for a in world.agents])
            l_pos = np.array([world.landmarks[0].state.p_pos for _ in range(self.num_agents)])
            reward_vec_dist = -np.sqrt(np.sum(np.square(l_pos - a_pos), axis=1))

            # reward 2 (agent collision)
            a_pos1 = np.array([a_pos]).repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = np.array([a_pos]).repeat(len(world.agents), axis=0)
            dist_a = np.sqrt(np.sum(np.square(a_pos1 - a_pos2), axis=2))
            reward_vec_coll = (dist_a < self.collision_threshold_agent).sum(axis=0) - 1

            reward_vec = reward_vec_dist - 3*reward_vec_coll

        return reward_vec

    def cost(self, agent, world):
        """
        Vectorized cost function
        """
        cost_vec = 0
        if agent == world.agents[0]:
            a_pos = np.array([a.state.p_pos for a in world.agents])
            l_pos = np.array([world.landmarks[0].state.p_pos for _ in range(self.num_agents)])
            dist_vec = np.sqrt(np.sum(np.square(a_pos - l_pos), axis=1))

            cost_vec = (dist_vec < self.collision_threshold)

        return cost_vec

    def observation(self, agent, world):
        entity_pos = [world.landmarks[0].state.p_pos - agent.state.p_pos]
        coll_dist = [[np.linalg.norm(entity_pos[0], 2) - self.collision_threshold]]

        # get positions of other agents in this agent's reference frame
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            relative_pos = other.state.p_pos - agent.state.p_pos
            other_pos.append(relative_pos)

        # choose closest other agents
        other_dist = np.sqrt(np.sum(np.square(np.array(other_pos)), axis=1))
        dist_idx = np.argsort(other_dist)
        other_pos = [other_pos[i] for i in dist_idx[:self.n_others]]

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + coll_dist + other_pos)

        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
