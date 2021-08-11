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

import pyglet
import pygame
import numpy as np
import Box2D
from Box2D.examples.framework import Framework, main, Keys
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2Random, b2Vec2, b2_dynamicBody, b2Color, b2_kinematicBody)
from pygame.locals import (QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN,
                           MOUSEBUTTONUP, MOUSEMOTION, KMOD_LSHIFT)

from cv.image_processor import ImageProcessor
from agent import Agent


ABSOLUTE_PATH = os.path.abspath('agi_proto')
HZ = 34


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
        self.hand = pygame.image.load(os.path.join(ABSOLUTE_PATH[:-14], 'pics', 'open.png')).convert_alpha()
        self.hand = pygame.transform.scale(self.hand, (self.hand.get_width() // 20, self.hand.get_height() // 20))
        self.hand_rect = self.hand.get_rect(topleft=(310, 400))
        self.min_ind = None

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

        bt = pygame.key.get_pressed()
        if bt[pygame.K_j]:
            self.hand_rect.centerx -= 5
            if self.hand_rect.left < 0:
                self.hand_rect.left = 0
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[0] -= 0.5

        elif bt[pygame.K_l]:
            self.hand_rect.centerx += 5
            if self.hand_rect.right > 639:
                self.hand_rect.right = 639
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[0] += 0.5

        elif bt[pygame.K_i]:
            self.hand_rect.centery -= 5
            if self.hand_rect.top < 0:
                self.hand_rect.top = 0
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[1] += 0.5

        elif bt[pygame.K_k]:
            self.hand_rect.centery += 5
            if self.hand_rect.bottom > 440:
                self.hand_rect.bottom = 440
            if self.min_ind:
                self.world.bodies[self.min_ind].worldCenter[1] -= 0.5

        if bt[pygame.K_q]:
            print('Turn on')
            if self.min_ind:
                self.world.bodies[self.min_ind].gravityScale = 1.0
                self.min_ind = None
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

        if bt[pygame.K_e]:
            if self.min_ind:
                direction_by_x = 0
                direction_by_y = 0
                if bt[pygame.K_w]:
                    direction_by_y += 10
                if bt[pygame.K_s]:
                    direction_by_y -= 10
                if bt[pygame.K_a]:
                    direction_by_x -= 10
                if bt[pygame.K_d]:
                    direction_by_x += 10
                self.world.bodies[self.min_ind].gravityScale = 1.0
                self.world.bodies[self.min_ind].linearVelocity[0] = direction_by_x
                self.world.bodies[self.min_ind].linearVelocity[1] = direction_by_y
                self.min_ind = None
        return True

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

            self.screen.blit(self.hand, self.hand_rect)
            pygame.image.save(pygame.display.get_surface(), 'rrrrr.png')
            # pygame.display.flip()
            pygame.display.update()
            clock.tick(HZ)
            self.fps = clock.get_fps()

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None


    def Print(self, str, color=(229, 153, 153, 255)):
        """
        Переопределили функцию которая делает тексты
        Draw some text at the top status lines
        and advance to the next line.
        """
        pass

agent = Agent()

class CollisionProcessing(CustomPygameFramework):

    last_step = None
    name = "CollisionProcessing"
    description = "Keys: left = a, right = d, down = s, up = w, grab = q, throw = e"
    x_offset = -10
    y_offset = 10
    grab = False
    ground_vertices = [(-50, 0), (50, 0)]
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
            shapes=b2EdgeShape(vertices=self.ground_vertices, )
        )

        xlow, xhi = -5, 5
        ylow, yhi = 2, 35
        random_vector = lambda: b2Vec2(b2Random(xlow, xhi), b2Random(ylow, yhi))

        x, y, z = 1.0, 200.0, 3.0
        c1 = b2Color(x, y, z)

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

    def get_image(self):
        buffer = pygame.PixelArray(self.screen)
        return buffer

    def Step(self, settings):

        # We are going to destroy some bodies according to contact
        # points. We must buffer the bodies that should be destroyed
        # because they may belong to multiple contact points.
        nuke = []

        self.world.bodies[-1].awake = True
        self.world.bodies[-2].awake = True

        #self.world.bodies[-1].inertia = 0.0
        # Traverse the contact results. Destroy bodies that
        # are touching heavier bodies.
        body_pairs = [(p['fixtureA'].body, p['fixtureB'].body)
                      for p in self.points]

        img_processor = ImageProcessor('rrrrr.png', arm_size=(self.hand_rect.topleft[0],
                                                              self.hand_rect.topleft[1],
                                                              self.hand_rect.size[0],
                                                              self.hand_rect.size[1]))
        self.cur_step = img_processor.run(self.last_step)
        self.last_step = [obj['center'] for obj in self.cur_step]
        agent.env_step(self.cur_step)

        #self.get_image()
        for body1, body2 in body_pairs:
            mass1, mass2 = body1.mass, body2.mass

            if mass1 > 0.0 and mass2 > 0.0:
                if mass2 > mass1:
                    nuke_body = body1
                else:
                    nuke_body = body2

                if nuke_body not in nuke:
                    nuke.append(nuke_body)
        nuke = []
        # Destroy the bodies, skipping duplicates.
        for b in nuke:
            print("Nuking:", b)
            self.world.DestroyBody(b)

        nuke = None


        super(CollisionProcessing, self).Step(settings)


if __name__ == "__main__":
    main(CollisionProcessing)
