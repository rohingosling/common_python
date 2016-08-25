
class Constant ( object ):

    class ID ( object ):

        class Component ( object ):
            UNIDENTIFIED = 0
            PHYSICS      = 1
            GEOMETRY     = 2
            STRUCTURE    = 3
            USER_INPUT   = 4

        class System ( object ):
            UNIDENTIFIED       = 0
            USER_INPUT_MANAGER = 1
            PHYSICS_SYSTEM     = 2
            COLLISION_DETECTOR = 3
            RENDERER           = 4

        class Entity ( object ):            
            UNIDENTIFIED = 0
            USER         = 1
            SHIP         = 2
            SHIP_SECTION = 3
            MISSILE      = 4
            BOARD        = 5

        class Group ( object ):            
            UNIDENTIFIED = 0
            USER         = 1
            SHIP         = 2
            SHIP_SECTION = 3
            MISSILE      = 4
            BOARD        = 5

    class Default ( object ):

        class Component ( object ):

            class Geometry ( object ):
                WIDTH  = 1
                HEIGHT = 3
                MARGIN = 1
                SCALE  = 1

        class Entity ( object ):
            
            class Ship ( object ):
                MIN_LENGTH = 3
                MAX_LENGTH = 6

