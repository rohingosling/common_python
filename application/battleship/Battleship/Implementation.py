###############################################################################
# Title:   Battleship
# Version: 1.0
# Date:    2016-08-18
# Author:  Rohin Gosling
#
# Description:
# - ECS (Entity Component Based) Battleship implementation, created for the 
#   solution to a Python Codecadamy project.
#
###############################################################################

import random
import math

from Platform    import Entity, Component, System, Application
from StringTable import StringTable
from Constant    import Constant
from GameMath    import GameMath


#------------------------------------------------------------------------------
# Components
#------------------------------------------------------------------------------

class Components ( object):

    class Physics ( Component ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id       = Constant.ID.Component.PHYSICS
            self.name     = StringTable.Implementation.Component.PHYSICS
            self.position = GameMath.Vector ( 0, 0 )
            self.heading  = GameMath.Vector ( 0, 0 )


    class Geometry ( Component ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id      = Constant.ID.Component.PHYSICS
            self.name    = StringTable.Implementation.Component.GEOMETRY
            self.width   = Constant.Default.Component.Geometry.WIDTH
            self.height  = Constant.Default.Component.Geometry.HEIGHT
            self.margin  = Constant.Default.Component.Geometry.MARGIN
            self.scale   = Constant.Default.Component.Geometry.SCALE
            self.aspect  = GameMath.Vector ( 1, 1 )


    class Structure ( Component ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id     = Constant.ID.Component.STRUCTURE
            self.name   = StringTable.Implementation.Component.STRUCTURE
            self.health = 1.0


    class UserInput ( Component ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id     = Constant.ID.Component.USER_INPUT
            self.name   = StringTable.Implementation.Component.USER_INPUT
            self.targit = GameMath.Vector ( 0, 0 )

#------------------------------------------------------------------------------
# Systems
#------------------------------------------------------------------------------

class Systems ( object ):

    class UserInputManager ( System ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id   = Constant.ID.System.USER_INPUT_MANAGER
            self.name = StringTable.Implementation.System.USER_INPUT_MANAGER

        def update(self):
            return super().update()
        
            # TODO: Implement system functionality.


    class PhysicsSystem ( System ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id   = Constant.ID.System.PHYSICS_SYSTEM
            self.name = StringTable.Implementation.System.PHYSICS_SYSTEM

        def update(self):
            return super().update()
    
            # TODO: Implement system functionality.


    class CollisionDetector ( System ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id   = Constant.ID.System.COLLISION_DETECTOR
            self.name = StringTable.Implementation.System.COLLISION_DETECTOR

        def update(self):
            return super().update()
    
            # TODO: Implement system functionality.


    class Renderer ( System ):

        def __init__ ( self ):
            Component.__init__ ( self )
            self.initialize ()

        def initialize ( self ):
            self.id   = Constant.ID.System.RENDERER
            self.name = StringTable.Implementation.System.RENDERER

        def update(self):
            return super().update()
    
            # TODO: Implement system functionality.

#------------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------------

class Battleship ( Application ):

    def __init__ ( self ):

        # Call base class constructor.

        Application.__init__ ( self )

        # Initalize derived class members.

        self.name = StringTable.Implementation.Application.NAME

        StringTable.Print.message ( 0, StringTable.Implementation.APPLICATION_STARTING )

        self.initialize_entities ()
        self.initialize_systems ()


    def initialize_entities ( self ):
        
        print ( "[APPLICATION]\t\tInitializing entities..." )

        # TODO: Add the strings abouve to the string table.
                
        # Configure Entity: Ship.

        entity = Entity ( Constant.ID.Entity.SHIP, StringTable.Implementation.Entity.SHIP, Constant.ID.Group.SHIP )
        entity.components.add_component ( Components.Physics () )
        entity.components.add_component ( Components.Geometry () )
        entity.components.add_component ( Components.Structure () )

        ship_length   = random.randint ( Constant.Default.Entity.Ship.MIN_LENGTH, Constant.Default.Entity.Ship.MAX_LENGTH )
        ship_position = GameMath.Vector ( 4, 4 )

        print ( "[APPLICATION]\t\tship_length = " + str (ship_length) )
        
        for section_index in range ( 0, ship_length ):
            sub_entity = Entity ( Constant.ID.Entity.SHIP_SECTION, StringTable.Implementation.Entity.SHIP_SECTION, Constant.ID.Group.SHIP_SECTION )
            sub_entity.components.add_component ( Components.Physics () )
            sub_entity.components.add_component ( Components.Geometry () )        
            sub_entity.components.add_component ( Components.Structure () )
            print ( "[APPLICATION]\t\tSub component #" + str ( section_index ) + ", added to parent entity." )

            entity.entities.add_entity ( sub_entity )

        # TO-DO: Add a function to retrieve a component, index or system from a list given their id.


        self.entities.add_entity ( entity )

        # Configure Entity: Missile.

        entity = Entity ( Constant.ID.Entity.MISSILE, StringTable.Implementation.Entity.MISSILE, Constant.ID.Group.MISSILE )
        entity.components.add_component ( Components.Physics () )
        self.entities.add_entity ( entity )

        # Configure Entity: User.

        entity = Entity ( Constant.ID.Entity.USER, StringTable.Implementation.Entity.USER, Constant.ID.Group.USER )
        entity.components.add_component ( Components.UserInput () )
        self.entities.add_entity ( entity )

        # Configure Entity: Game board.  

        entity = Entity ( Constant.ID.Entity.BOARD, StringTable.Implementation.Entity.BOARD, Constant.ID.Group.BOARD )
        entity.components.add_component ( Components.Geometry () )
        self.entities.add_entity ( entity )

    def initialize_systems ( self ):
        
        print ( "[APPLICATION]\t\tInitializing systems..." )

        # TODO: Add the strings abouve to the string table.

        self.systems.add_system ( Systems.UserInputManager() )
        self.systems.add_system ( Systems.PhysicsSystem() )
        self.systems.add_system ( Systems.CollisionDetector() )
        self.systems.add_system ( Systems.Renderer() )
        


#------------------------------------------------------------------------------
# Program Entry Point:
#------------------------------------------------------------------------------

def main():

    application = Battleship()    
    application.run ()
    
main()

#------------------------------------------------------------------------------
# ...
#------------------------------------------------------------------------------

