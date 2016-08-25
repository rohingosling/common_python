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

from StringTable import StringTable
from Constant    import Constant

class ApplicationBase ( object ):
    
    def __init__ ( self ):
        self.id    = Constant.ID.UNIDENTIFIED
        self.name  = StringTable.Message.NO_DATA
        self.group = Constant.ID.UNIDENTIFIED


#------------------------------------------------------------------------------
# Components
#------------------------------------------------------------------------------

class Component ( ApplicationBase ):

    def __init__ ( self ):
        self.id    = Constant.ID.Component.UNIDENTIFIED
        self.name  = StringTable.Platform.COMPONENT
        self.group = Constant.ID.Group.UNIDENTIFIED

    #///////////////////////////////////////////////////////////////
    # TODO: Move ptint_info out of the manager class, and into here.
    #///////////////////////////////////////////////////////////////

class ComponentManager ( object ):

    def __init__ ( self ):
        self.components = []

    def add_component ( self, component ):
        self.components.append ( component )
        self.print_info ( component )

    def print_info ( self, component ):
        s =  StringTable.Platform.COMPONENT_MANAGER
        s += StringTable.Message.TAB
        s += "Initializing component;\t"
        s += "id=" + str ( component.id )
        s += ",\t"
        s += "name=" + StringTable.Function.quote_double ( component.name )       
        print ( s )


#------------------------------------------------------------------------------
# Entities
#------------------------------------------------------------------------------

class Entity ( ApplicationBase ):

    def __init__ ( self, id, name, group ):
        self.id         = id
        self.name       = name
        self.group      = group
        self.components = ComponentManager ()
        self.entities   = EntityManager ()

    #///////////////////////////////////////////////////////////////
    # TODO: Move ptint_info out of the manager class, and into here.
    #///////////////////////////////////////////////////////////////


class EntityManager ( object ):
    
    def __init__ ( self ):        
        self.entities = []

    def add_entity ( self, entity ):
        self.entities.append ( entity )
        self.print_info ( entity )

    def print_info ( self, entity ):
        s =  StringTable.Platform.ENTITY_MANAGER
        s += StringTable.Message.TAB
        s += "Initializing entity;\t"
        s += "id=" + str ( entity.id )
        s += ",\t"
        s += "name=" + StringTable.Function.quote_double ( entity.name )
        s += ",\t"
        s += "group=" + str ( entity.group )        
        print ( s )

#------------------------------------------------------------------------------
# Systems
#------------------------------------------------------------------------------

class System ( ApplicationBase ):

    def __init__ ( self, id, name, group ):
        self.id    = id
        self.name  = name
        self.group = group

    def update ( self ):
        pass

    #///////////////////////////////////////////////////////////////
    # TODO: Move ptint_info out of the manager class, and into here.
    #///////////////////////////////////////////////////////////////


class SystemManager ( object ):
    
    def __init__ ( self ):        
        self.systems = []

    def add_system ( self, system ):
        self.systems.append ( system )
        self.print_info ( system )

    def print_info ( self, system ):
        s =  StringTable.Platform.SYSTEM_MANAGER
        s += StringTable.Message.TAB
        s += "Initializing system;\t"
        s += "id=" + str ( system.id )
        s += ",\t"
        s += "name=" + StringTable.Function.quote_double ( system.name )        
        print ( s )


#------------------------------------------------------------------------------
# Message
#------------------------------------------------------------------------------

class Message ( ApplicationBase ):

    def __init__ ( self, id, name, group ):
        self.id    = id
        self.name  = name
        self.group = group


class MessageQueue ( object ):
    
    def __init__ ( self ):        
        self.message_queue = []

    def post_message ( self, system ):
        self.message_queue.append ( system )


#------------------------------------------------------------------------------
# Application
#------------------------------------------------------------------------------

class Application ( ApplicationBase ):

    def __init__ ( self ):
        self.entities      = EntityManager ()
        self.systems       = SystemManager ()
        self.message_queue = MessageQueue ()        
        StringTable.Print.message ( 0, StringTable.Platform.PROGRAM_STARTING )

    def run ( self ):
        StringTable.Print.message ( 0, StringTable.Platform.PROGRAM_TERMINATED )
        StringTable.Print.new_line ()


#------------------------------------------------------------------------------
# ...
#------------------------------------------------------------------------------
    
