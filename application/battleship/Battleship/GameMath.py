###############################################################################
# Title:   GameMath
# Version: 1.0
# Date:    2016-08-18
# Author:  Rohin Gosling
#
# Description:
# - General purpose game and graphics related math functions.
#
###############################################################################

class GameMath ( object ):

    #--------------------------------------------------------------------------
    # Vector
    #--------------------------------------------------------------------------

    class Vector ( object ):
    
        def __init__ ( self, x, y ):
            self.assign ( x, y )

        def assign ( self, x, y ):
            self.x = x
            self.y = y

        def magnitude ( self ):
            x = self.x
            y = self.y
            m = math.fabs ( math.sqrt ( x*x + y*y ) )
            return m

        def add ( self, vector ):
            x = self.x + vector.x
            y = self.y + vector.y
            return GameMath.Vector ( x, y )

        def subtract ( self, vector ):
            x = self.x - vector.x
            y = self.y - vector.y
            return GameMath.Vector ( x, y )

        def scale ( self, factor ):
            x = self.x * factor
            y = self.y * factor
            return GameMath.Vector ( x, y )

        def hadamard_product ( self, vector ):
            x = self.x * vector.x
            y = self.y * vector.y
            return GameMath.Vector ( x, y )

        def flip ( self ):
            x = -self.x
            y = -self.y
            return ( x, y )

        def to_string ( self ):

            output_string  = Utility.StringTable.PERENTHESUS_OPEN
            output_string += str ( self.x )
            output_string += Utility.StringTable.DELIMINATOR
            output_string += str ( self.y )
            output_string += Utility.StringTable.PERENTHESUS_CLOSE
            
            return output_string
