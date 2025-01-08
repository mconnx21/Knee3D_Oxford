#ASSUME WORKING IN THE SAGGITAL AXIS SCHEME DEFINTED BY THE BOUNDING BOX GIVEN IN LOADEDDICOMS

#INPUT: two points given as [x1,y,z1] and [x2,y,z2] in our defined coordinate system. Note same y coordiante
#OUTOUT: a list of coordinates (in plane Y = y) which are a midpoint algorithm representation of the line between the two input points
import matplotlib.pyplot as plt
def linePixels2D(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    assert(y1 == y2)
    line = []

    #now will start midpoint algorithm 
    dz = z2 -z1
    dx = x2-x1


    def f(x,z):
        return dz*x - z*dx + z1*x2 - z2*x1

    #lots of cases to work through
    if x2 == x1:
        if z1 < z2:
            for i in range (0, z2-z1):
                line.append([x1, z1+ i ])
            return line
        else:
            #wont be same as otherwise same points so z2 < z1
            for i in range (0, z1-z2):
                line.append([x1, z1-i])
            return line
    
    elif z2 == z1:
        if x1 < x2:
            for i in range (0, x2-x1):
                line.append([x1 + i, z1])
            return line
        else:
            #wont be same as otherwise same points so z2 < z1
            for i in range (0, x1-x2):
                line.append([x1 - i, z1])
            return line
    
    elif x1 < x2 and z1 < z2:
        #then positive gradiant in first first quater
        m = dz / dx
        if m <1 :
            #then midpoint alg will consider steps of (x+1, z) and (x+1, z+1)

            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx < x2 and currentz < z2:
                if fval + 1*dz - 0.5*dx < 0:
                    #then midpoint is over the line, so better of taking a shallower step
                    currentx = currentx +1
                    fval = fval + dz
                    
                else:
                    currentx = currentx + 1
                    currentz = currentz + 1
                    fval = fval + dz -dx   
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line
                
        else:
            #then alg will consider steps of (x, z+1) and (x+1, z+1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx < x2 and currentz < z2:
                if fval + 0.5*dz - 1*dx > 0:
                    #then midpoint is under the line, so better of taking a steeper step
                    currentz = currentz + 1
                    fval = fval - dx
                else:
                    currentx = currentx +1
                    currentz = currentz +1
                    fval = fval + dz -dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line

    elif x1 < x2 and z1 > z2:
        #then neg gradiant fourth quarter
        m = abs(dz) /dx
        if m <1:
            #then alg will consider steps of (x+1, z) and (x+1, z-1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx < x2 and currentz > z2:
                if fval + 1*dz + 0.5*dx > 0:
                    #then midpoint is below the line (neg grad line down), so better of taking a shallower step
                    currentx = currentx + 1
                    fval = fval + dz
                else:
                    currentx = currentx +1
                    currentz = currentz -1
                    fval = fval + dz +dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line
        else:
            #then alg wiill consider steps of (x, z-1) and (x+1, z-1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx < x2 and currentz > z2:
                if fval + 0.5*dz + 1*dx < 0:
                    #then midpoint is above the line (neg grad line down), so better of taking a steeper step
                    currentz = currentz - 1
                    fval = fval + dx
                else:
                    currentx = currentx +1
                    currentz = currentz -1
                    fval = fval + dz +dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line

    elif x1 > x2 and z1 < z2:
        #neg gradient second quarter
        m = dz / abs(dx)
        if m <1:
            #then alg will consider steps of (x-1, z) and (x-1, z+1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx > x2 and currentz < z2:
                if fval - 1*dz - 0.5*dx > 0:
                    #then midpoint is above the line, so better of taking a shallower step
                    currentx = currentx - 1
                    fval = fval - dz
                else:
                    currentx = currentx -1
                    currentz = currentz +1
                    fval = fval - dz -dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line
        else:
            #then alg wiill consider steps of (x, z+1) and (x-1, z+1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx > x2 and currentz < z2:
                if fval - 0.5*dz - 1*dx < 0:
                    #then midpoint is below the line, so better of taking a steeper step
                    currentz = currentz + 1
                    fval = fval - dx
                else:
                    currentx = currentx -1
                    currentz = currentz +1
                    fval = fval - dz -dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line
    elif x1 > x2 and z1> z2:
        #pos gradient thrid quarter
        m = abs(dz)/abs(dx)
        if m <1:
            #then alg will consider steps of (x-1, z) and (x-1, z-1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx > x2 and currentz > z2:
                if fval - 1*dz + 0.5*dx < 0:
                    #then midpoint is bellow the line, so better of taking a shallower step
                    currentx = currentx - 1
                    fval = fval - dz
                else:
                    currentx = currentx -1
                    currentz = currentz -1
                    fval = fval - dz +dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line
        else:
            #then alg wiill consider steps of (x, z-1) and (x-1, z-1)
            fval = 0
            currentx = x1
            currentz = z1
            line.append([currentx, currentz])
            while currentx > x2 and currentz > z2:
                if fval - 0.5*dz + 1*dx > 0:
                    #then midpoint is above the line, so better of taking a steeper step
                    currentz = currentz - 1
                    fval = fval +dx
                else:
                    currentx = currentx -1
                    currentz = currentz -1
                    fval = fval - dz +dx
                line.append([currentx, currentz])
            #note as it stands, x2,z2 may not actually end up on the line, but it will be closest fit
            return line
        

"""
line = linePixels2D([5,2, 0], [-308,2, 0])
xs = [x[0] for x in line]
ys = [x[1] for x in line]
plt.plot(xs,ys)
plt.show()
"""
            


        

    

