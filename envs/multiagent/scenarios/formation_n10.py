"""
Formation: N agents are tasked to form a circular formation
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from learning.envs.multiagent.core_vec import World, Agent, Landmark
from learning.envs.multiagent.scenario import BaseScenario


def get_thetas(poses):
    # compute angles of the agents
    num_agent = len(poses)
    thetas = [None]*num_agent
    for i in range(num_agent):
        pose = poses[i]
        angle = np.arctan2(pose[1], pose[0])
        if angle < 0:
            angle += 2*np.pi
        thetas[i] = angle

    return thetas


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 10
        num_landmarks = 1
        # world.collaborative = True
        self.num_agents = num_agents
        self.circle_radius = 0.7
        self.world_radius = 0.9
        self.n_others = 4
        self.np_rnd = np.random.RandomState(0)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.idx = i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05

        # add landmarks (a target location)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03

        # make initial conditions
        ideal_theta_separation = (2*np.pi)/num_agents
        self.initial_config = np.transpose([self.circle_radius*np.array((np.cos(i*ideal_theta_separation), np.sin(i*ideal_theta_separation)))
                                            for i in range(num_agents)])
        self.collide_th = 2*world.agents[0].size
        self.mean_pos = np.zeros((world.dim_p,))
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
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.world_radius, self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # bound on the landmark position less than that of the environment for visualization purposes
            landmark.state.p_pos = np.random.uniform(-0.5*self.world_radius, 0.5*self.world_radius, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.mean_pos = np.zeros((world.dim_p,))

    def reward(self, agent, world):
        rew = 0
        if agent == world.agents[0]:
            total_pos = np.array([agent.state.p_pos for agent in world.agents])
            self.mean_pos = np.mean(total_pos, axis=0)
            relative_poses = [agent.state.p_pos - self.mean_pos for agent in world.agents]
            thetas = get_thetas(relative_poses)
            theta_min = np.min(thetas)
            rotation_matrix = np.array([[np.cos(theta_min), -np.sin(theta_min)], [np.sin(theta_min), np.cos(theta_min)]])
            expected_poses = np.transpose(np.dot(rotation_matrix, self.initial_config)) + self.mean_pos
            expected_poses = np.array([expected_poses]).repeat(len(world.agents), axis=0)
            agent_poses1 = np.array([total_pos]).repeat(len(world.agents), axis=0)
            agent_poses2 = np.transpose(agent_poses1, axes=(1, 0, 2))
            dists = np.sqrt(np.sum(np.square(agent_poses2 - expected_poses), axis=2))
            row_ind, col_ind = linear_sum_assignment(dists)
            rew -= dists[row_ind, col_ind].sum()

            dist_a = np.sqrt(np.sum(np.square(agent_poses1 - agent_poses2), axis=2))
            n_collide = (dist_a < self.collide_th).sum() - len(world.agents)
            rew -= n_collide

        rew_vec = [rew]*self.num_agents

        return rew_vec

    def cost(self, agent, world):
        cost = 0
        if agent == world.agents[0]:
            cost = np.linalg.norm(self.mean_pos - world.landmarks[0].state.p_pos, 2)

        cost_vec = [cost]*self.num_agents

        return cost_vec

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = [world.landmarks[0].state.p_pos - self.mean_pos]
        formation_dist = [[np.linalg.norm(entity_pos[0], 2)]]

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # choose closest other agents
        other_dist = np.sqrt(np.sum(np.square(np.array(other_pos)), axis=1))
        dist_idx = np.argsort(other_dist)
        other_pos = [other_pos[i] for i in dist_idx[:self.n_others]]

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos + formation_dist)

        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
