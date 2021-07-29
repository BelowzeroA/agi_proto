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

import pyglet
import pygame
import numpy as np
import Box2D
from Box2D.examples.framework import Framework, main, Keys
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2Random, b2Vec2, b2_dynamicBody, b2Color, b2_kinematicBody)
import cv2 as cv

from cv_play import main_2, main_3
from agent import Agent


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
            for ind in range(len(self.test.world.bodies)):
                u = our_zoom(self.test.world.bodies[ind].worldCenter)
                v = center_of_mass(vertices)
                if self.EPS > abs(u[0] - v[0]) and self.EPS > abs(u[0] - v[0]):
                    color = self.test.our_color[ind]
            pygame.draw.polygon(self.surface, color, vertices, 0)

            #print(our_zoom(self.test.world.bodies[5].worldCenter))
            #print(center_of_mass(vertices))
            pygame.draw.polygon(self.surface, b2Color(0.9, 0.7, 0.7), vertices, 1)

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
                           center, radius, 0)
        pygame.draw.circle(self.surface, b2Color(0.9, 0.7, 0.7), center, radius, 1)
        pygame.draw.aaline(self.surface, (255, 0, 0), center,
                           (center[0] - radius * axis[0],
                            center[1] + radius * axis[1]))

    def DrawCircle(self, center, radius, color, drawwidth=1):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        #print(len(self.test.world.bodies))
        #print(self.test.world.bodies[0])
        #for ind in range(len(self.test.world.bodies)):
        #    u = our_zoom(self.test.world.bodies[ind].worldCenter)
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
                           center, radius, drawwidth)


class CustomPygameFramework(Box2D.examples.backends.pygame_framework.PygameFramework):

    def __init__(self):
        super().__init__()
        self.renderer = CustomDraw(surface=self.screen, test=self)
        #print(self.setCenter(self.world.bodies[0].worldCenter))
        self.world.renderer = self.renderer

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

            pygame.display.flip()
            pygame.image.save(pygame.display.get_surface(), 'rrrrr.png')
            #pygame.image.save(self.screen, '..rrrrr.png')
            clock.tick(self.settings.hz)
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


class CollisionProcessing(CustomPygameFramework):
    agent = Agent()
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
        #CustomPygameFramework.__init__(self)


        # Tell the framework we're going to use contacts, so keep track of them
        # every Step.
        self.using_contacts = True


        # Ground body
        world = self.world
        ground = world.CreateBody(
            shapes=b2EdgeShape(vertices=self.ground_vertices,)
        )

        xlow, xhi = -5, 5
        ylow, yhi = 2, 35
        random_vector = lambda: b2Vec2(b2Random(xlow, xhi), b2Random(ylow, yhi))

        x, y, z = 1.0, 200.0, 3.0
        c1 = b2Color(x, y, z)
        #""" Для того чтоб остались одни шары
        # Small triangle
        triangle = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(-6, 0), (2, 0), (0, 4)]),
            density=1,
        )

        world.CreateBody(
            type=b2_dynamicBody,
            #position=random_vector(),
            position=(2, 0),
            fixtures=triangle,
        )

        # Small circle
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1,
        )


        # Large circle
        circle.shape.radius *= 2
        world.CreateBody(
            type=b2_dynamicBody,
            #position=random_vector(),
            position=(10, 0),
            fixtures=circle,
        )

        vertices = [(0, 0), (-4, 0), (-2, 1), ]
        vertices = [(2 * x, 2 * y) for x, y in vertices]
        triangle = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=10,
            restitution=1
        )

        triangle_left = world.CreateBody(
            type=b2_dynamicBody,
            #type=b2_kinematicBodyз,
            position=(self.x_offset, self.y_offset),
            fixtures=triangle,
            gravityScale=0.0,
            awake=True,
        )
        triangle_left.My_color = (1, 1, 1)



    def Keyboard(self, key):
        try:
            if key == Keys.K_a:
                self.world.bodies[-1].worldCenter[0] -= 1
                if self.grab:
                    self.world.bodies[-1].contacts[0].contact.fixtureA.body.worldCenter[0] -= 1
            elif key == Keys.K_s:
                self.world.bodies[-1].worldCenter[1] -= 1
                if self.grab:
                    self.world.bodies[-1].contacts[0].contact.fixtureA.body.worldCenter[1] -= 1
            elif key == Keys.K_d:
                self.world.bodies[-1].worldCenter[0] += 1
                if self.grab:
                    self.world.bodies[-1].contacts[0].contact.fixtureA.body.worldCenter[0] += 1
            elif key == Keys.K_w:
                self.world.bodies[-1].worldCenter[1] += 1
                if self.grab:
                    self.world.bodies[-1].contacts[0].contact.fixtureA.body.worldCenter[1] += 1
            elif key == Keys.K_q:
                if self.grab:
                    self.grab = False
                    self.world.bodies[-1].contacts[0].contact.fixtureA.body.gravityScale = 1.0
                else:
                    self.grab = True
                    self.world.bodies[-1].contacts[0].contact.fixtureA.body.gravityScale = 0.0
            elif key == Keys.K_e:
                my_force = 5
                delta_x = (self.world.bodies[-1].contacts[0].contact.fixtureA.body.worldCenter[0] -
                           self.world.bodies[-1].worldCenter[0])
                delta_y = (self.world.bodies[-1].contacts[0].contact.fixtureA.body.worldCenter[1] -
                           self.world.bodies[-1].worldCenter[1])
                self.grab = False
                self.world.bodies[-1].contacts[0].contact.fixtureA.body.gravityScale = 1.0
                self.world.bodies[-1].contacts[0].contact.fixtureA.body.linearVelocity[0] = my_force * delta_x
                self.world.bodies[-1].contacts[0].contact.fixtureA.body.linearVelocity[1] = - my_force * delta_y
        except IndexError:
            pass

    def get_image0(self):
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(buffer.height, buffer.width, 4)
        return arr

    def get_image(self):
        # return
        buffer = pygame.PixelArray(self.screen)
        # buffer[10:15, :, :]
        # return buffer
        return buffer

    def Step(self, settings):

        # We are going to destroy some bodies according to contact
        # points. We must buffer the bodies that should be destroyed
        # because they may belong to multiple contact points.
        nuke = []
        self.world.bodies[-1].awake = True
        self.world.bodies[-1].linearVelocity[0] = 0
        self.world.bodies[-1].linearVelocity[1] = 0
        self.world.bodies[-1].angularVelocity = 0
        #self.world.bodies[-1].inertia = 0.0
        # Traverse the contact results. Destroy bodies that
        # are touching heavier bodies.
        body_pairs = [(p['fixtureA'].body, p['fixtureB'].body)
                      for p in self.points]

        main_3('rrrrr.png', agent=self.agent)
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
