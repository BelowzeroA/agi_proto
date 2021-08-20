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

import pygame
import numpy as np
import Box2D
from Box2D.examples.framework import Framework, main, Keys
from Box2D import (b2CircleShape, b2FixtureDef, b2PolygonShape, b2LoopShape,
                   b2Random, b2Vec2, b2_dynamicBody, b2Color)
from pygame.locals import (QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN,
                           MOUSEBUTTONUP, MOUSEMOTION, KMOD_LSHIFT)

from cv.image_processor import ImageProcessor
from agent import Agent
from utils import path_from_root


HZ = 34


agent = Agent()


try:
    from .pygame_gui import (fwGUI, gui)
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

            # print(our_zoom(self.test.world.bodies[5].worldCenter))
            # print(center_of_mass(vertices))
            # pygame.draw.polygon(self.surface, b2Color(0.9, 0.7, 0.7), vertices, 1)

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        for ind in range(len(self.test.world.bodies)):
            u = our_zoom(self.test.world.bodies[ind].worldCenter)
            if self.EPS > abs(u[0] - center[0]) and self.EPS > abs(u[0] - center[0]):
                cur_ind = ind
                color = self.test.our_color[ind]

        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(self.surface, color,
                           center, radius - 3, 0)


class CustomPygameFramework(Box2D.examples.backends.pygame_framework.PygameFramework):

    def __init__(self):
        super().__init__()
        self.renderer = CustomDraw(surface=self.screen, test=self)
        self.world.renderer = self.renderer
        self.hand = pygame.image.load(path_from_root('pics/open.png')).convert_alpha()
        self.hand = pygame.transform.scale(self.hand, (self.hand.get_width() // 20, self.hand.get_height() // 20))
        self.hand_close = pygame.image.load(path_from_root('pics/open.png')).convert_alpha()
        self.hand_close = pygame.transform.scale(self.hand_close, (self.hand_close.get_width() // 20,
                                                                   self.hand_close.get_height() // 20))
        self.hand_push = pygame.image.load(path_from_root('pics/push.png')).convert_alpha()
        self.hand_push = pygame.transform.scale(self.hand_push, (self.hand_push.get_width() // 25,
                                                                   self.hand_push.get_height() // 25))
        self.hand_push_r = self.hand_push
        self.hand_push_l = pygame.transform.flip(self.hand_push, 1, 0)
        self.hand_rect = self.hand.get_rect(topleft=(410, 350))
        self.min_ind = None
        self.push = None
        self.pixel_array = None
        self.arm_step = {'right': 0, 'up': 0}
        self.f_sys = pygame.font.SysFont('arial', 12)


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

    def checkEvents(self):
        """
        Check for pygame events (mainly keyboard/mouse events).
        Passes the events onto the GUI also.
        """
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == Keys.K_ESCAPE):
                return False
            elif event.type == KEYDOWN:
                self._Keyboard_Event(event.key, down=True)
            elif event.type == KEYUP:
                self._Keyboard_Event(event.key, down=False)
            elif event.type == MOUSEBUTTONDOWN:
                p = self.ConvertScreenToWorld(*event.pos)
                if event.button == 1:  # left
                    mods = pygame.key.get_mods()
                    if mods & KMOD_LSHIFT:
                        self.ShiftMouseDown(p)
                    else:
                        self.MouseDown(p)
                elif event.button == 2:  # middle
                    pass
                elif event.button == 3:  # right
                    self.rMouseDown = True
                elif event.button == 4:
                    self.viewZoom *= 1.1
                elif event.button == 5:
                    self.viewZoom /= 1.1
            elif event.type == MOUSEBUTTONUP:
                p = self.ConvertScreenToWorld(*event.pos)
                if event.button == 3:  # right
                    self.rMouseDown = False
                else:
                    self.MouseUp(p)
            elif event.type == MOUSEMOTION:
                p = self.ConvertScreenToWorld(*event.pos)

                self.MouseMove(p)

                if self.rMouseDown:
                    self.viewCenter -= (event.rel[0] /
                                        5.0, -event.rel[1] / 5.0)

            if GUIEnabled:
                self.gui_app.event(event)  # Pass the event to the GUI

        right = 0
        up = 0
        rotation = 0

        bt = pygame.key.get_pressed()
        if bt[pygame.K_f]:
            if self.push:
                self.push = False
                self.hand_rect = self.hand.get_rect(center=self.hand_rect.center)
            else:
                self.push = 1
                self.hand_rect = self.hand_push.get_rect(center=self.hand_rect.center)

        if bt[pygame.K_e]:
            if self.min_ind:
                self.world.bodies[self.min_ind].gravityScale = 1.0
                self.world.bodies[self.min_ind].linearVelocity[0] = self.arm_step['right']
                self.world.bodies[self.min_ind].linearVelocity[1] = self.arm_step['up']
                self.min_ind = None

        if bt[pygame.K_a] or agent.actions['move_left']:
            k = 1 if agent.actions['move_left'] == 2 else 0.5
            self.hand_rect.centerx -= 5 * k
            right -= 10 * k
            if self.hand_rect.left < 0:
                self.hand_rect.left = 0
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[0] -= 0.5 * k
            elif self.push:
                self.hand_push = self.hand_push_l
                self.push_near_object(val=25)

        if bt[pygame.K_d] or agent.actions['move_right']:
            k = 1 if agent.actions['move_right'] == 2 else 0.5
            self.hand_rect.centerx += 5 * k
            right += 10 * k
            if self.hand_rect.right > 639:
                self.hand_rect.right = 639
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[0] += 0.5 * k
            elif self.push:
                self.hand_push = self.hand_push_r
                self.push_near_object(val=25)

        if bt[pygame.K_w] or agent.actions['move_up']:
            k = 1 if agent.actions['move_up'] == 2 else 0.5
            self.hand_rect.centery -= 5 * k
            up += 10 * k
            if self.hand_rect.top < 255:
                self.hand_rect.top = 255
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[1] += 0.5 * k

        if bt[pygame.K_s] or agent.actions['move_down']:
            k = 1 if agent.actions['move_down'] == 2 else 0.5
            self.hand_rect.centery += 5 * k
            up -= 10 * k
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

        if bt[pygame.K_q]:
            if self.min_ind:
                pass
            else:
                list_ind = []
                for ind in range(len(self.world.bodies)):
                    u = our_zoom(self.world.bodies[ind].worldCenter)
                    dist = (abs(self.hand_rect.center[0] - u[0]) +
                            abs(self.hand_rect.center[1] - u[1]))
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
            self.hand_rect = self.hand_close.get_rect(center=self.hand_rect.center)

        self.arm_step['right'] = right
        self.arm_step['up'] = up
        return True

    def Print(self, str="", color=(229, 153, 153, 255)):
        sc_text = self.f_sys.render('Surprise: %s' % (agent.surprise), 1, color, (0, 0, 0))
        text_pos = sc_text.get_rect(topleft=(10, 30))
        self.screen.blit(sc_text, text_pos)
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
            if self.min_ind:
                self.screen.blit(self.hand_close, self.hand_rect)
            elif self.push:
                self.screen.blit(self.hand_push, self.hand_rect)
            else:
                self.screen.blit(self.hand, self.hand_rect)
            self.pixel_array = self.get_imag(pygame.display.get_surface())
            pygame.display.update()
            clock.tick(HZ)
            self.fps = clock.get_fps()

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None


class CollisionProcessing(CustomPygameFramework):
    num_step = 0
    last_step = None
    name = "CollisionProcessing"
    description = "Keys: left = a, right = d, down = s, up = w, grab = q, throw = e"
    x_offset = -10
    y_offset = 10
    grab = False
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
        super(CollisionProcessing, self).__init__()
        self.using_contacts = True

        # Ground body
        world = self.world
        ground = world.CreateBody(
            shapes=b2LoopShape(vertices=self.ground_vertices, )
        )

        # xlow, xhi = -5, 5
        # ylow, yhi = 2, 35

        # x, y, z = 1.0, 200.0, 3.0
        # c1 = b2Color(x, y, z)

        circle = b2FixtureDef(
            shape=b2CircleShape(radius=2),
            density=1,
            restitution=0.5
        )

        world.CreateBody(
            type=b2_dynamicBody,
            position=(10, 0),
            fixtures=circle,
            awake=True,
        )

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
        )
        triangle_left.My_color = (1, 1, 1)

    def Keyboard(self, key):
        pass

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

    def Step(self, settings):
        # We are going to destroy some bodies according to contact
        # points. We must buffer the bodies that should be destroyed
        # because they may belong to multiple contact points.

        self.world.bodies[-1].awake = True
        self.world.bodies[-2].awake = True

        if np.all(self.pixel_array != None): #and self.num_step == 5:
            self.num_step = 0
            img_processor = ImageProcessor(self.pixel_array, arm_size=(self.hand_rect.topleft[0],
                                                                       self.hand_rect.topleft[1],
                                                                       self.hand_rect.size[0],
                                                                       self.hand_rect.size[1]))
            self.cur_step = img_processor.run(self.last_step)
            self.last_step = [obj['center'] for obj in self.cur_step]
            agent.env_step(self.cur_step)

        self.num_step += 1

        super(CollisionProcessing, self).Step(settings)



if __name__ == "__main__":
    main(CollisionProcessing)
