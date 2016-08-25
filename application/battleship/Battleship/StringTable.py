
class StringTableBase ( object ):
    
    class Print ( object ):

        def new_line ():
            print ( StringTable.Message.EMPTY )

        def message ( tab, msg ):
            t = StringTable.Message.SPACE * tab            
            print ( t + msg )

    class Function ( object ):

        def bracket_round ( s ):

            return "(" + s + ")"

        def bracket_square ( s ):

            return "[" + s + "]"

        def bracket_angle ( s ):

            return "<" + s + ">"

        def quote_double ( s ):

            return "\"" + s + "\""

        def quote_single ( s ):

            return "'" + s + "'"

class StringTable ( StringTableBase ):

    class Message ( object ):
        EMPTY                  = ""
        SPACE                  = " "
        NEW_LINE               = "\n"
        NO_DATA                = "-"
        TAB                    = "\t"

    class Platform ( object ):
        TAB                    = "\t"        
        COMPONENT              = "Component"
        COMPONENT_MANAGER      = StringTableBase.Function.bracket_square ( "COMPONENT_MANAGER" )
        ENTITY_MANAGER         = StringTableBase.Function.bracket_square ( "ENTITY_MANAGER" )
        SYSTEM_MANAGER         = StringTableBase.Function.bracket_square ( "SYSTEM_MANAGER" )
        PLATFORM               = StringTableBase.Function.bracket_square ( "PLATFORM" )
        PROGRAM_TERMINATED     = PLATFORM + (TAB * 3) + "ECS platform terminated."
        PROGRAM_STARTING       = PLATFORM + (TAB * 3) + "ECS platform starting."

    class Implementation ( object ):
        TAB                    = "\t"
        APPLICATION            = StringTableBase.Function.bracket_square ( "APPLICATION" )
        APPLICATION_TERMINATED = APPLICATION + (TAB * 2) + "Application terminated."
        APPLICATION_STARTING   = APPLICATION + (TAB * 2) + "Application starting."

        class Application ( object ):
            NAME = "Battleship"
                
        class Component ( object ):            
            PHYSICS    = "Physics properties"
            GEOMETRY   = "Geometric properties"
            STRUCTURE  = "Structural properties"
            USER_INPUT = "User input data"

        class Entity ( object ):
            UNIDENTIFIED = "Unidentified Entity"
            USER         = "User"
            SHIP         = "Ship"
            SHIP_SECTION = "Ship Section"
            MISSILE      = "Missile"
            BOARD        = "Board"

        class System ( object ):            
            USER_INPUT_MANAGER = "User input manager"
            PHYSICS_SYSTEM     = "Physics system"
            COLLISION_DETECTOR = "Collision detection system"
            RENDERER           = "Renderer"

    class Test ( object ):
        HELLO_WORLD = "Hello World!"



