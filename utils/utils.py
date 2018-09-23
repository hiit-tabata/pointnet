import os 

def savePoints(filename, points):
    """
    write the pointes to a file
    params:
        filename: string asb path
        points: npArray nbPts:n
    """    
    with open(filename, 'w') as file:
        for row in points:
            for col in row:
                file.write(str(col)+' ')
            file.write('\n')
    os.chmod(filename, 0o777)
    
