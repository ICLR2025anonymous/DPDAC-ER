"""
Scenario: A position swapping task with velocity boundedness constraints
Note: In the individual observation, the relative position lists are sorted.
"""
import numpy as np
from learning.envs.multiagent.core_vec import World, Agent, Landmark
from learning.envs.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 10
        num_landmarks = 10
        self.num_agents = num_agents
        self.n_others = 5
        # world.collaborative = True
        self.world_radius = 1.2  # camera range
        self.np_rnd = np.random.RandomState(0)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.idx = i
            agent.collide = False
            agent.silent = True
            agent.size = 0.06
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.collision_threshold = 2*world.agents[0].size
        self.velocity_bound = 1.1
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
        radius_all = 0.8 + np.random.uniform(-0.05, 0.05)
        angle_all = np.random.uniform(-np.pi, np.pi)

        rand = np.random.uniform(0, 1)
        for i in range(self.num_agents):
            if rand <= 0.5:
                theta_i = i*(2*np.pi/self.num_agents) + angle_all
            else:
                theta_i = -i*(2*np.pi/self.num_agents) + angle_all

            p_x = radius_all*np.cos(theta_i)
            p_y = radius_all*np.sin(theta_i)

            # agents
            world.agents[i].state.p_pos = np.array((p_x, p_y))
            world.agents[i].state.p_vel = np.zeros(world.dim_p)
            world.agents[i].state.c = np.zeros(world.dim_c)

            # corresponding landmarks
            j = int(np.mod((i + self.num_agents/2), self.num_agents))
            world.landmarks[j].state.p_pos = np.array((p_x, p_y)) + np.random.uniform(-0.05, 0.05, 2)
            world.landmarks[j].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """
        Vectorized reward function
        """
        reward_vec = 0
        if agent == world.agents[0]:
            # reward 1 (target)
            a_pos = np.array([a.state.p_pos for a in world.agents])
            l_pos = np.array([l.state.p_pos for l in world.landmarks])
            reward_vec_dist = -np.sqrt(np.sum(np.square(l_pos - a_pos), axis=1))

            # reward 2 (agent collision)
            a_pos1 = np.array([a_pos]).repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = np.array([a_pos]).repeat(len(world.agents), axis=0)
            dist_a = np.sqrt(np.sum(np.square(a_pos1 - a_pos2), axis=2))
            reward_vec_coll = (dist_a < self.collision_threshold).sum(axis=0) - 1

            reward_vec = reward_vec_dist - reward_vec_coll

        return reward_vec

    def cost(self, agent, world):
        """
        Vectorized cost function
        """
        cost_vec = 0
        if agent == world.agents[0]:
            a_vel = np.array([a.state.p_vel for a in world.agents])
            a_vel_scalar = np.sqrt(np.sum(np.square(a_vel), axis=1))
            cost_vec = (a_vel_scalar > self.velocity_bound)

        return cost_vec

    def observation(self, agent, world):
        idx = agent.idx
        entity_pos = [world.landmarks[idx].state.p_pos - agent.state.p_pos]
        entity_dist = [[np.linalg.norm(entity_pos[0], 2)]]

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

        obs = np.concatenate(entity_pos + entity_dist + [agent.state.p_vel] + [agent.state.p_pos] + other_pos
                             + [[np.linalg.norm(agent.state.p_vel, 2)]])

        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
