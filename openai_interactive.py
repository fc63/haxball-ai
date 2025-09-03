import os
import sys
import time
from argparse import Namespace

import pygame

from simulator import create_start_conditions, Vector
# Prefer Python engine here to guarantee latest physics behavior without requiring a Cython rebuild
from simulator.visualizer import draw_frame

# Defer heavy TF/baselines imports to runtime to avoid spawn-time failures on Windows
USE_MODEL = True
try:
    import numpy as np
    from hx_controller.haxball_gym import Haxball
    from hx_controller.haxball_vecenv import HaxballProcPoolVecEnv
    from baselines.common.policies import build_policy
    from hx_controller.openai_model_torneo import A2CModel
except Exception as e:
    print("[interactive] Model disabled due to import error:", e)
    USE_MODEL = False


class DelayedModel:
    def __init__(self, env: Haxball, model: A2CModel, play_red: bool) -> None:
        self.state = 0
        self.env = env
        self.model = model
        self.play_red = play_red
        self.wait_time = 2

    def gameplay_tick(self):
        if self.state == 0:
            # Prendiamo obs
            self.obs, self.rew, self.done, self.info = self.env.step_wait(red_team=not play_red)

            # print(obs)
            reward = self.rew
            if self.done:
                env.reset()

            self.state = 2
            self.wait_time = 0

        elif self.state == 1:
            # Aspettiamo un po'
            if self.wait_time == 0:
                self.state = 2
            self.wait_time -= 1

        elif self.state == 2:
            # Facciamo una predizione
            self.actions, self.rew, _, _ = self.model.step(np.array([self.obs]), M=[self.done], S=None)
            self.state = 4
            self.wait_time = 0

        elif self.state == 3:
            # Aspettiamo un po'
            if self.wait_time == 0:
                self.state = 4
            self.wait_time -= 1

        elif self.state == 4:
            ret = float(self.rew[0])
            # actions, rew, _, _ = model.step(obs, S=None, M=[0])
            action = self.actions[0]

            self.env.step_async(action, red_team=not self.play_red)
            self.state = 0
            self.wait_time = 5

        elif self.state == 5:
            # Aspettiamo un po'
            if self.wait_time == 0:
                self.state = 0
            self.wait_time -= 1


if __name__ == '__main__':
    args_namespace = Namespace(
        alg='a2c',
        env='haxball-v0',
        num_env=None,
        # env='PongNoFrameskip-v4',
        env_type=None, gamestate=None, network=None, num_timesteps=1000000.0, play=False,
        reward_scale=1.0, save_path=None, save_video_interval=0, save_video_length=200, seed=None)
    # env2 = build_env(args_namespace)

    try:
        from mpi4py import MPI  # type: ignore
    except ImportError:
        MPI = None
    from baselines import logger

    # Optional model setup
    env = None
    model = None
    max_ticks = int(60*3*(1/0.016))
    if USE_MODEL:
        try:
            nsteps = 3
            nenvs = 2
            total_timesteps = int(15e7)
            # load_path can be overridden; default below may not exist
            load_path = 'ppo2_best_so_far2.h5'
            env = HaxballProcPoolVecEnv(num_fields=nenvs, max_ticks=max_ticks)
            policy = build_policy(env=env, policy_network='mlp', num_layers=4, num_hidden=256)
            model = A2CModel(policy, model_name='ppo_model_0', env=env, nsteps=nsteps, ent_coef=0.05, total_timesteps=total_timesteps, lr=7e-4)
            if load_path is not None and os.path.exists(load_path):
                model.load(load_path)
        except Exception as e:
            print("[interactive] Disabling model at runtime:", e)
            USE_MODEL = False
    # model = StaticModel()
    # model = RandomModel(action_space=env.action_space)
    # model = PazzoModel(action_space=env.action_space)
    # model = StaticModel(default_action=7, action_space=env.action_space)
    # model = StaticModel(action_space=env.action_space)
    # model = MoreRealisticModel(action_space=env.action_space)
    # nbatch = 100 * 12
    # nbatch_train = nbatch // 4
    # model = PPOModel(policy=policy, nsteps=12, ent_coef=0.05, ob_space=env.observation_space, ac_space=env.action_space, nbatch_act=100, nbatch_train=nbatch_train, vf_coef=0.5, max_grad_norm=0.5)# 0.005) #, vf_coef=0.0)


    size = width, height = 900, 520
    center = (width // 2, height // 2 + 30)
    black = 105, 150, 90

    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)

    gameplay = create_start_conditions(
        posizione_palla=Vector(0, 0),
        velocita_palla=Vector(0, 0),
        posizione_blu=Vector(277.5, 0),
        velocita_blu=Vector(0, 0),
        input_blu=0,
        posizione_rosso=Vector(-277.5, 0),
        velocita_rosso=Vector(0, 0),
        input_rosso=0,
        tempo_iniziale=0,
        punteggio_rosso=0,
        punteggio_blu=0
    )

    if USE_MODEL and env is not None:
        env = Haxball(gameplay=gameplay, max_ticks=max_ticks*2)
        obs = env.reset()
        action = 0
    play_red = 0

    dm = DelayedModel(env, model, play_red) if (USE_MODEL and model is not None and env is not None) else None

    blue_unpressed = True
    red_unpressed = True

    D_i = 1 if play_red else 2
    i = 0
    reward = None
    ret = None
    next_action = 0
    while True:
        i += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        gameplay.Pa.D[D_i].mb = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            gameplay.Pa.D[D_i].mb |= 1
        if keys[pygame.K_DOWN]:
            gameplay.Pa.D[D_i].mb |= 2
        if keys[pygame.K_RIGHT]:
            gameplay.Pa.D[D_i].mb |= 8
        if keys[pygame.K_LEFT]:
            gameplay.Pa.D[D_i].mb |= 4
        if keys[pygame.K_SPACE]:
            # Keep kick bit on while held for slowdown; use edge to trigger impulse.
            if blue_unpressed:
                gameplay.Pa.D[D_i].mb |= 16  # rising edge triggers impulse inside engine
                blue_unpressed = False
            # Maintain hold by keeping bit set while pressed
            gameplay.Pa.D[D_i].mb |= 16
            gameplay.Pa.D[D_i].bc = 1
        else:
            gameplay.Pa.D[D_i].bc = 0
            blue_unpressed = True

        # a1, a2 = data
        # env.step_async(a1, red_team=True)
        if dm is not None:
            dm.gameplay_tick()

        # obs = env.get_observation(action)
        # actions = model.step(obs)

        # gameplay.Pa.D[1].mb = 0
        # gameplay.Pa.D[1].bc = 0
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_w]:
        #     gameplay.Pa.D[1].mb |= 1
        # if keys[pygame.K_s]:
        #     gameplay.Pa.D[1].mb |= 2
        # if keys[pygame.K_d]:
        #     gameplay.Pa.D[1].mb |= 8
        # if keys[pygame.K_a]:
        #     gameplay.Pa.D[1].mb |= 4
        # if keys[pygame.K_LCTRL]:
        #     if red_unpressed:
        #         gameplay.Pa.D[1].mb |= 16
        #         gameplay.Pa.D[1].bc = 1
        #     red_unpressed = False
        # else:
        #     red_unpressed = True

        draw_frame(screen, gameplay, reward=reward, ret=ret)

        # screen.blit(ball, ballrect)
        pygame.display.flip()
        clock.tick(60)
        gameplay.step(1)

        # if gameplay.zb == 2:
        #     gameplay = create_start_conditions(
        #         posizione_palla=Vector(0, 0),
        #         velocita_palla=Vector(0, 0),
        #         posizione_blu=Vector(277.5, 0),
        #         velocita_blu=Vector(0, 0),
        #         input_blu=0,
        #         posizione_rosso=Vector(-277.5, 0),
        #         velocita_rosso=Vector(0, 0),
        #         input_rosso=0,
        #         tempo_iniziale=gameplay.Ac,
        #         punteggio_rosso=gameplay.Kb,
        #         punteggio_blu=gameplay.Cb,
        #         commincia_rosso=gameplay.Jd.sn == 't-red'
        #     )