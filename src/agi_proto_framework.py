#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version by Ken Lauer / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

import os
import time
from argparse import ArgumentParser

import pygame
import numpy as np
import Box2D
from Box2D.examples.framework import Framework, Keys
from Box2D import (b2CircleShape, b2FixtureDef, b2PolygonShape, b2LoopShape,
                   b2Random, b2Vec2, b2_dynamicBody, b2Color)
from Box2D.examples.settings import fwSettings
from pygame.locals import (QUIT, KEYDOWN)
from cv.image_processor import ImageProcessor
from agent import Agent
from utils import path_from_root

from agent_hand import AgentHand
from main import SERVER, DISTANCE


HZ = 34


agent = Agent()



try:
    from backends.pygame_gui import (fwGUI, gui)
    GUIEnabled = True
except Exception as ex:
    print('Unable to load PGU; menu disabled.')
    print('(%s) %s' % (ex.__class__.__name__, ex))
    GUIEnabled = False


def center_of_mass(vertices):
    x = 0
    y = 0
    for line in vertices:
        x += line[0]
        y += line[1]
    return x / len(vertices), y / len(vertices)


def our_zoom(vertices):
    x = vertices[0]
    y = vertices[1]
    return x * 10 + 320, (20 - y) * 10 + 240


class CustomDraw(Box2D.examples.backends.pygame_framework.PygameDraw):

    EPS = 1

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the specified color.
        """
        if not vertices:
            return
        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color, vertices[0], vertices[1])
        else:
            v = center_of_mass(vertices)
            for ind in range(len(self.test.world.bodies)):
                u = our_zoom(self.test.world.bodies[ind].worldCenter)
                if self.EPS > abs(u[0] - v[0]) and self.EPS > abs(u[0] - v[0]):
                    color = self.test.our_color[ind]
            temp = []
            for p in vertices:
                if p[1] - v[1] < 0:
                    if p[0] - v[0] < 0:
                        temp.append((p[0] + 3, p[1] + 3))
                    else:
                        temp.append((p[0] - 3, p[1] + 3))
                else:
                    if p[0] - v[0] < 0:
                        temp.append((p[0] + 3, p[1] - 4))
                    else:
                        temp.append((p[0] - 3, p[1] - 4))
            pygame.draw.polygon(self.surface, color, temp, 0)

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        for ind in range(len(self.test.world.bodies)):
            u = our_zoom(self.test.world.bodies[ind].worldCenter)
            if self.EPS > abs(u[0] - center[0]) and self.EPS > abs(u[0] - center[0]):
                color = self.test.our_color[ind]

        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(self.surface, color, center, radius - 3, 0)


class CustomPygameFramework(Box2D.examples.framework.FrameworkBase if SERVER
                            else Box2D.examples.backends.pygame_framework.PygameFramework,
                            ImageProcessor):

    def __init__(self):
        super().__init__()
        if not SERVER:
            self.renderer = CustomDraw(surface=self.screen, test=self)
            self.hand = pygame.image.load(path_from_root('pics/open.png')).convert_alpha()
            self.hand = pygame.transform.scale(self.hand, (self.hand.get_width() // 20, self.hand.get_height() // 20))
            self.hand_close = pygame.image.load(path_from_root('pics/close1.png')).convert_alpha()
            self.hand_close = pygame.transform.scale(self.hand_close, (self.hand_close.get_width() // 20,
                                                                       self.hand_close.get_height() // 20))
            self.hand_close.set_colorkey((255, 255, 255))
            self.hand_push = pygame.image.load(path_from_root('pics/push.png')).convert_alpha()
            self.hand_push = pygame.transform.scale(self.hand_push, (self.hand_push.get_width() // 25,
                                                                     self.hand_push.get_height() // 25))
            self.hand_push_r = self.hand_push
            self.hand_push_l = pygame.transform.flip(self.hand_push, 1, 0)
            self.hand_rect = self.hand.get_rect(topleft=(410, 350))
            # self.hand_rect = self.hand.get_rect(topleft=(410, 403))
            #self.hand_rect = self.hand.get_rect(topleft=(389, 410))
            self.f_sys = pygame.font.SysFont('arial', 12)
        self.world.renderer = self.renderer
        self.min_ind = None
        self.grab = False
        self.push = None
        self.pixel_array = None
        self.arm_step = {'right': 0, 'up': 0}
        self.attention = None
        self.pixel_array = None
        self.pi_alpha = np.pi / 180
        self.popo = 0

    def push_near_object(self, val=15):
        for ind in range(len(self.world.bodies)):
            u = our_zoom(self.world.bodies[ind].worldCenter)
            dist = (abs(self.hand_rect.center[0] - u[0]) +
                    abs(self.hand_rect.center[1] - u[1]))
            if dist < val:
                dist = self.hand_rect.center[0] - u[0]
                if dist > 0:
                    self.world.bodies[ind].linearVelocity[0] = -15
                else:
                    self.world.bodies[ind].linearVelocity[0] = 15
                self.world.bodies[ind].linearVelocity[1] = 3

    @staticmethod
    def set_body_dynamics(body, gravity=0, velocity_x=0, velocity_y=0):
        body.gravityScale = gravity
        body.linearVelocity[0] = velocity_x
        body.linearVelocity[1] = velocity_y

    def checkEvents(self):
        """
        Check for pygame events (mainly keyboard/mouse events).
        Passes the events onto the GUI also.
        """
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == Keys.K_ESCAPE):
                return False

        right = 0
        up = 0
        rotation = 0
        contact_approx_contour = None

        if self.min_ind:
            if type(self.world.bodies[self.min_ind].fixtures[0].shape) == Box2D.b2CircleShape:
                approx_contour = self.get_approx_for_circle(self.world.bodies[self.min_ind])
            elif type(self.world.bodies[self.min_ind].fixtures[0].shape) == Box2D.Box2D.b2PolygonShape:
                approx_contour = self.get_approx_for_poly(self.world.bodies[self.min_ind].fixtures[0].shape.vertices,
                                                          self.world.bodies[self.min_ind].transform.angle,
                                                          self.world.bodies[self.min_ind].transform.position)
            approx_contour = [our_zoom(p) for p in approx_contour]

        bt = pygame.key.get_pressed()

        if bt[pygame.K_f]:
            if self.push:
                self.push = False
                self.hand_rect = self.hand.get_rect(center=self.hand_rect.center)
            else:
                self.push = 1
                self.hand_rect = self.hand_push.get_rect(center=self.hand_rect.center)

        # if bt[pygame.K_e]:
        #     if self.min_ind:
        #         self.world.bodies[self.min_ind].gravityScale = 1.0
        #         self.world.bodies[self.min_ind].linearVelocity[0] = self.arm_step['right']
        #         self.world.bodies[self.min_ind].linearVelocity[1] = self.arm_step['up']
        #         self.min_ind = None

        if bt[pygame.K_a] or agent.actions['move_left']:
            k = 1 if agent.actions['move_left'] == 2 else 0.5
            right -= 10 * k
            if not (self.min_ind and len([p[0] for p in approx_contour if p[0] < 3]) != 0):
                self.hand_rect.centerx -= 5 * k
                if self.hand_rect.left < 0:
                    self.hand_rect.left = 0
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[0] -= 0.5 * k
                elif self.push:
                    self.hand_push = self.hand_push_l
                    self.push_near_object(val=25)

        if bt[pygame.K_d] or agent.actions['move_right']:
            k = 1 if agent.actions['move_right'] == 2 else 0.5
            right += 10 * k
            if not (self.min_ind and len([p[0] for p in approx_contour if p[0] > 636]) != 0):
                self.hand_rect.centerx += 5 * k
                if self.hand_rect.right > 639:
                    self.hand_rect.right = 639
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[0] += 0.5 * k
                elif self.push:
                    self.hand_push = self.hand_push_r
                    self.push_near_object(val=25)

        if bt[pygame.K_w] or agent.actions['move_up']:
            k = 1 if agent.actions['move_up'] == 2 else 0.5
            up += 10 * k
            if not (self.min_ind and len([p[1] for p in approx_contour if p[1] < 255]) != 0):
                self.hand_rect.centery -= 5 * k
                if self.hand_rect.top < 255:
                    self.hand_rect.top = 255
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[1] += 0.5 * k

        if bt[pygame.K_s] or agent.actions['move_down']:
            k = 1 if agent.actions['move_down'] == 2 else 0.5
            up -= 10 * k
            if not (self.min_ind and len([p[1] for p in approx_contour if p[1] > 435]) != 0):
                self.hand_rect.centery += 5 * k
                if self.hand_rect.bottom > 437:
                    self.hand_rect.bottom = 437
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[1] -= 0.5 * k

        if bt[pygame.K_m]:
            rotation -= 5
            if self.min_ind:
                #self.world.bodies[self.min_ind].angle -= 0.5
                self.world.bodies[self.min_ind].setTransform(0.5)

        if bt[pygame.K_n]:
            rotation += 5
            if self.min_ind:
                self.world.bodies[self.min_ind].angle += 0.5

        if self.grab and self.min_ind and agent.actions['grab'] == 0:
            body = self.world.bodies[self.min_ind]
            self.set_body_dynamics(body, 1.0, self.arm_step['right'], self.arm_step['up'])
            self.min_ind = None
            self.grab = False

        if self.grab and self.min_ind and agent.actions['grab'] == 1:
            body = self.world.bodies[self.min_ind]
            self.set_body_dynamics(body)

        elif self.grab and not self.min_ind and agent.actions['grab'] == 0:
            self.grab = False

        elif not self.grab and agent.actions['grab'] == 1:
            self.grab = True
            list_ind = []
            for ind in range(len(self.world.bodies)):
                u = our_zoom(self.world.bodies[ind].worldCenter)
                dist = (abs(self.hand_rect.center[0] - u[0]) + abs(self.hand_rect.center[1] - u[1]))
                if dist < DISTANCE:
                    list_ind.append((ind, dist))
            if len(list_ind) > 0:
                self.min_ind = 0
                for ind in range(len(list_ind)):
                    if list_ind[self.min_ind][1] > list_ind[ind][1]:
                        self.min_ind = ind
                self.min_ind = list_ind[self.min_ind][0]
                body = self.world.bodies[self.min_ind]
                self.set_body_dynamics(body)

        self.arm_step['right'] = right
        self.arm_step['up'] = up
        return True

    def Print(self, my_str="", color=(229, 153, 153, 255)):
        sc_text = self.f_sys.render(
            'Surprise: %s' % self.agent_message['surprise'], 1, color, (0, 0, 0)
        )
        sc_text_2 = self.f_sys.render(
            'Current tick: %s' % self.agent_message['current_tick'], 1, color, (0, 0, 0)
        )
        text_3 = self.f_sys.render(f'energy: {agent.striatum_energy:3.1f}', 1, color, (0, 0, 0))

        text_pos = sc_text.get_rect(topleft=(13, 13))
        text_pos_2 = sc_text.get_rect(topleft=(13, 33))
        text_pos_3 = sc_text.get_rect(topleft=(100, 33))
        self.screen.blit(sc_text, text_pos)
        self.screen.blit(sc_text_2, text_pos_2)
        self.screen.blit(text_3, text_pos_3)
        """
        Переопределили функцию которая делает тексты
        Draw some text at the top status lines
        and advance to the next line.
        """

    def run(self):
        """
        Main loop.
        Continues to run while checkEvents indicates the user has
        requested to quit.
        Updates the screen and tells the GUI to paint itself.
        """
        # If any of the test constructors update the settings, reflect
        # those changes on the GUI before running
        if GUIEnabled:
            self.gui_table.updateGUI(self.settings)

        running = True
        clock = pygame.time.Clock()
        while running:
            running = self.checkEvents()
            self.screen.fill((0, 0, 0))

            # Check keys that should be checked every loop (not only on initial
            # keydown)
            self.CheckKeys()

            # Run the simulation loop
            self.SimulationLoop()

            if GUIEnabled and self.settings.drawMenu:
                self.gui_app.paint(self.screen)

            self.Print()
            if self.grab:
                self.screen.blit(self.hand_close, self.hand_rect)
            elif self.push:
                self.screen.blit(self.hand_push, self.hand_rect)
            else:
                self.screen.blit(self.hand, self.hand_rect)
            attention_point = [-1, -1]
            #self.pixel_array = self.get_imag(pygame.display.get_surface())
            self.pixel_array = self.screen
            if (self.agent_message and self.agent_message['attention-spot']['x'] != -1 and
                    self.agent_message['attention-spot']['y'] != -1):
                attention_point[0] = self.agent_message['attention-spot']['x']
                attention_point[1] = self.agent_message['attention-spot']['y']
                pygame.draw.circle(self.screen, (255, 0, 0), attention_point, 30, 1)
            pygame.display.update()
            clock.tick(HZ)
            self.fps = clock.get_fps()

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None


class AgiProtoFramework(Box2D.examples.framework.FrameworkBase if SERVER else CustomPygameFramework):
    arm_step = {'right': 0, 'up': 0}
    # settings = sfwSettings
    if SERVER:
        agent_hand = AgentHand()
        agent_hand += (9.1, 6)
    num_step = 0
    last_step = None
    last_data = None
    # print(os.environ)
    agent_message = {'surprise': 0, 'current_tick': 0, 'attention-spot': {'x': -1, 'y': -1}}

    name = "AGI proto"
    description = "Keys: left = a, right = d, down = s, up = w, grab = q, throw = e"
    x_offset = -10
    y_offset = 10
    grab = False
    min_ind = False
    push = False
    ground_vertices = [(-32, 38), (-32, 0), (32, 0), (32, 38)]
    our_color = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (100, 0, 100),
        (0, 100, 100),
        (150, 100, 0),
        (100, 100, 100),
        (50, 100, 150),
        (250, 100, 50),
        (0, 250, 100)
    ]

    def __init__(self):
        super(AgiProtoFramework, self).__init__()
        self.using_contacts = True

        # Ground body
        world = self.world
        ground = world.CreateBody(
            shapes=b2LoopShape(vertices=self.ground_vertices, )
        )

        self.current_tick = 0
        self.init_circle(world)
        self.init_triangle(world)
        self.init_teacher(world)
        # self.init_teacher2(world)

    def init_circle(self, world):
        fixture = b2FixtureDef(
            shape=b2CircleShape(radius=2),
            density=1,
            restitution=0.5
        )

        circle = world.CreateBody(
            type=b2_dynamicBody,
            position=(10, 2),
            fixtures=fixture,
            awake=True,
            userData='circle'
        )

    def init_triangle(self, world):
        vertices = [(0, 0), (-4, 0), (-2, 2), ]
        vertices = [(2 * x, 2 * y) for x, y in vertices]
        triangle = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=10,
            restitution=0.2
        )

        triangle_left = world.CreateBody(
            type=b2_dynamicBody,
            position=(self.x_offset, 0),
            fixtures=triangle,
            gravityScale=1.0,
            awake=True,
            userData='triangle'
        )
        triangle_left.My_color = (1, 1, 1)

    def init_teacher(self, world):
        triangle = b2FixtureDef(
            # shape=b2PolygonShape(vertices=vertices),
            shape=b2PolygonShape(box=(1.5, 1.5)),
            density=10,
            restitution=0.2
        )
        triangle.name = 'Teacher'
        teacher = world.CreateBody(
            type=b2_dynamicBody,
            position=(self.x_offset + 20, 20),
            fixtures=triangle,
            gravityScale=0.0,
            awake=True,
            userData='Teacher'
        )
        teacher.My_color = (1, 2, 1)

    def Keyboard(self):
        right = 0
        up = 0
        rotation = 0

        if self.min_ind:
            if type(self.world.bodies[self.min_ind].fixtures[0].shape) == Box2D.b2CircleShape:
                approx_contour = self.get_approx_for_circle(self.world.bodies[self.min_ind])
            elif type(self.world.bodies[self.min_ind].fixtures[0].shape) == Box2D.Box2D.b2PolygonShape:
                approx_contour = self.get_approx_for_poly(self.world.bodies[self.min_ind].fixtures[0].shape.vertices,
                                                          self.world.bodies[self.min_ind].transform.angle,
                                                          self.world.bodies[self.min_ind].transform.position)
            approx_contour = [our_zoom(p) for p in approx_contour]

        if self.grab and self.min_ind and agent.actions['grab'] == 0:
            self.world.bodies[self.min_ind].gravityScale = 1.0
            self.world.bodies[self.min_ind].linearVelocity[0] = self.arm_step['right']
            self.world.bodies[self.min_ind].linearVelocity[1] = self.arm_step['up']
            self.min_ind = None
            self.grab = False

        elif self.grab and not self.min_ind and agent.actions['grab'] == 0:
            self.grab = False

        elif not self.grab and agent.actions['grab'] == 1:
            self.grab = True
            list_ind = []
            for ind in range(len(self.world.bodies)):
                u = our_zoom(self.world.bodies[ind].worldCenter)
                dist = (abs(self.agent_hand._center[0] - u[0]) +
                        abs(self.agent_hand._center[1] - u[1]))
                if dist < 20:
                    list_ind.append((ind, dist))
            if len(list_ind) > 0:
                self.min_ind = 0
                for ind in range(len(list_ind)):
                    if list_ind[self.min_ind][1] > list_ind[ind][1]:
                        self.min_ind = ind
                self.min_ind = list_ind[self.min_ind][0]
                self.world.bodies[self.min_ind].gravityScale = 0.0
                self.world.bodies[self.min_ind].linearVelocity[0] = 0
                self.world.bodies[self.min_ind].linearVelocity[1] = 0

        # if agent.actions['grab']:
        #     if self.grab and self.min_ind:
        #         self.world.bodies[self.min_ind].gravityScale = 1.0
        #         self.world.bodies[self.min_ind].linearVelocity[0] = self.arm_step['right']
        #         self.world.bodies[self.min_ind].linearVelocity[1] = self.arm_step['up']
        #         self.min_ind = None
        #         self.grab = False
        #     elif self.grab and not self.min_ind:
        #         self.grab = False
        #     else:
        #         self.grab = True
        #         list_ind = []
        #         for ind in range(len(self.world.bodies)):
        #             u = our_zoom(self.world.bodies[ind].worldCenter)
        #             dist = (abs(self.hand_rect.center[0] - u[0]) +
        #                     abs(self.hand_rect.center[1] - u[1]))
        #             if dist < 20:
        #                 list_ind.append((ind, dist))
        #         if len(list_ind) > 0:
        #             self.min_ind = 0
        #             for ind in range(len(list_ind)):
        #                 if list_ind[self.min_ind][1] > list_ind[ind][1]:
        #                     self.min_ind = ind
        #             self.min_ind = list_ind[self.min_ind][0]
        #             self.world.bodies[self.min_ind].gravityScale = 0.0
        #             self.world.bodies[self.min_ind].linearVelocity[0] = 0
        #             self.world.bodies[self.min_ind].linearVelocity[1] = 0

        if agent.actions['move_left']:
            k = 1 if agent.actions['move_left'] == 2 else 0.5
            right -= 10 * k
            if not (self.min_ind and len([p[0] for p in approx_contour if p[0] < 3]) != 0):
                if not self.agent_hand.left < -31.9:
                    self.agent_hand -= (0.5 * k, 0)
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[0] -= 0.5 * k
                elif self.push:
                    self.hand_push = self.hand_push_l
                    self.push_near_object(val=25)

        if agent.actions['move_right']:
            k = 1 if agent.actions['move_right'] == 2 else 0.5
            right += 10 * k
            if not (self.min_ind and len([p[0] for p in approx_contour if p[0] > 636]) != 0):
                if not self.agent_hand.right > 31.9:
                    self.agent_hand += (0.5 * k, 0)
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[0] += 0.5 * k
                elif self.push:
                    self.hand_push = self.hand_push_r
                    self.push_near_object(val=25)

        if agent.actions['move_up']:
            k = 1 if agent.actions['move_up'] == 2 else 0.5
            up += 10 * k
            if not (self.min_ind and len([p[1] for p in approx_contour if p[1] < 255]) != 0):
                if not self.agent_hand.top > 16:
                    self.agent_hand += (0, 0.5 * k)
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[1] += 0.5 * k

        if agent.actions['move_down']:
            k = 1 if agent.actions['move_down'] == 2 else 0.5
            up -= 10 * k
            if not (self.min_ind and len([p[1] for p in approx_contour if p[1] > 435]) != 0):
                if not self.agent_hand.bottom < 0.1:
                    self.agent_hand -= (0, 0.5 * k)
                if self.min_ind:
                    self.world.bodies[self.min_ind].worldCenter[1] -= 0.5 * k

        self.arm_step['right'] = right
        self.arm_step['up'] = up
        return True

    def get_imag(self, pixel_array):
        buffer = pygame.PixelArray(pixel_array)
        res = np.zeros((480, 640, 4), dtype="uint8")
        for i in range(buffer.shape[1]):
            for j in range(buffer.shape[0]):
                res[i, j, 3] = pygame.Color(buffer[j, i]).r
                res[i, j, 2] = pygame.Color(buffer[j, i]).g
                res[i, j, 1] = pygame.Color(buffer[j, i]).b
                res[i, j, 0] = pygame.Color(buffer[j, i]).a
        return res

    def get_image(self, pixel_array):
        buffer = pygame.surfarray.array3d(pixel_array)
        return buffer

    def viewing_the_status(self):
        grab_dict = {'is_clenched': False,
                     'is_holding': False}
        if self.min_ind:
            grab_dict['is_holding'] = True
        if self.grab:
            grab_dict['is_clenched'] = True
        self.cur_step = {
            'data': self.cur_step,
            'mode': grab_dict
        }

    def Step(self, settings):
        for body in self.world.bodies:
            if body.userData:
                body.awake = True

        if (not SERVER) and np.all(self.pixel_array is not None):
            img_processor = ImageProcessor(self.world, SERVER, self.pixel_array, arm_size=(
                self.hand_rect.topleft[0],
                self.hand_rect.topleft[1],
                self.hand_rect.size[0],
                self.hand_rect.size[1]))
            self.cur_step = img_processor.run(self.last_step, self.last_data)
            self.last_step = [obj['center'] for obj in self.cur_step]
            self.last_data = self.cur_step
            self.viewing_the_status()
            # time.sleep(10)
            self.agent_message = agent.env_step(self.cur_step)
            self.current_tick = self.agent_message['current_tick']

        elif SERVER:
            img_processor = ImageProcessor(self.world,
                                           SERVER,
                                           arm_size=(
                                               self.agent_hand.left,
                                               self.agent_hand.top,
                                               self.agent_hand.right - self.agent_hand.left,
                                               self.agent_hand.top - self.agent_hand.bottom),
                                           arm=self.agent_hand.hand_contour)
            self.cur_step = img_processor.run(self.last_step, self.last_data)
            self.last_step = [obj['center'] for obj in self.cur_step]
            self.last_data = self.cur_step
            self.viewing_the_status()
            self.agent_message = agent.env_step(self.cur_step)
            self.Keyboard()

        self.num_step += 1

        super(AgiProtoFramework, self).Step(settings)
