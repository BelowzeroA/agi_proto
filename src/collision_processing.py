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
from Box2D.examples.framework import Framework, main
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2Random, b2Vec2, b2_dynamicBody, b2Color)


class CollisionProcessing (Framework):
    name = "CollisionProcessing"

    def __init__(self):
        super(CollisionProcessing, self).__init__()

        # Tell the framework we're going to use contacts, so keep track of them
        # every Step.
        self.using_contacts = True

        # Ground body
        world = self.world
        ground = world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-50, 0), (50, 0)],)
        )

        xlow, xhi = -5, 5
        ylow, yhi = 2, 35
        random_vector = lambda: b2Vec2(b2Random(xlow, xhi), b2Random(ylow, yhi))

        x, y, z = 1.0, 200.0, 3.0
        c1 = b2Color(x, y, z)
        # Small triangle
        triangle = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(-3, 0), (1, 0), (0, 2)]),
            density=1,
        )

        world.CreateBody(
            type=b2_dynamicBody,
            position=random_vector(),
            fixtures=triangle,
        )

        body = world.CreateDynamicBody(position=(2, 4))
        box = body.CreatePolygonFixture(box=(2, 2), density=1, friction=0.3)

        # Large triangle (recycle definitions)
        triangle.shape.vertices = [
            2.0 * b2Vec2(v) for v in triangle.shape.vertices]

        tri_body = world.CreateBody(type=b2_dynamicBody,
                                    position=random_vector(),
                                    fixtures=triangle,
                                    fixedRotation=True,  # <--
                                    )
        # note that the large triangle will not rotate

        # Small box
        box = b2FixtureDef(
            shape=b2PolygonShape(box=(1, 0.5)),
            density=1,
            restitution=0.1,
        )

        world.CreateBody(
            type=b2_dynamicBody,
            position=random_vector(),
            fixtures=box,
        )

        # Large box
        box.shape.box = (2, 1)
        world.CreateBody(
            type=b2_dynamicBody,
            position=random_vector(),
            fixtures=box,
        )

        # Small circle
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=1),
            density=1,
        )

        world.CreateBody(
            type=b2_dynamicBody,
            position=random_vector(),
            fixtures=circle,
        )

        # Large circle
        circle.shape.radius *= 2
        world.CreateBody(
            type=b2_dynamicBody,
            position=random_vector(),
            fixtures=circle,
        )

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

    def Step(self, settings):
        # We are going to destroy some bodies according to contact
        # points. We must buffer the bodies that should be destroyed
        # because they may belong to multiple contact points.
        nuke = []

        # Traverse the contact results. Destroy bodies that
        # are touching heavier bodies.
        body_pairs = [(p['fixtureA'].body, p['fixtureB'].body)
                      for p in self.points]
        self.get_image()

        for body1, body2 in body_pairs:
            mass1, mass2 = body1.mass, body2.mass

            if mass1 > 0.0 and mass2 > 0.0:
                if mass2 > mass1:
                    nuke_body = body1
                else:
                    nuke_body = body2

                if nuke_body not in nuke:
                    nuke.append(nuke_body)

        # Destroy the bodies, skipping duplicates.
        for b in nuke:
            print("Nuking:", b)
            self.world.DestroyBody(b)

        nuke = None

        super(CollisionProcessing, self).Step(settings)


if __name__ == "__main__":
    main(CollisionProcessing)
