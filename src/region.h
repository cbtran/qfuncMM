#ifndef REGION_H
#define REGION_H

#define MAX_VOXEL 1000

class VOXEL {
  
private:
  int x, y, z;
public:
  VOXEL(int xLoc=0, int yLoc=0, int zLoc=0): x(xLoc), y(yLoc), z(zLoc) {};
  ~VOXEL() {};
  
  int getX() const {
    return x;
  }
  
  int getY() const {
    return y;
  }
  
  int getZ() const {
    return z;
  }
  
  void setX(int xLoc) {
    x = xLoc;
    return;
  }
  
  void setY(int yLoc) {
    y = yLoc;
    return;
  }
  
  void setZ(int zLoc) {
    z = zLoc;
    return;
  }
};

class REGION {
    private:
       int size;
       VOXEL region[MAX_VOXEL];
    public:
      REGION(int input) {
        assert(pow(input, 3) <= MAX_VOXEL);
        
        size = input;
        
        int voxelID = 0;
        
        for (int x = 1; x <= size; x++) {
          for (int y = 1; y <= size; y++) {
            for (int z = 1; z <= size; z++) {
              
              region[voxelID].setX(x);
              region[voxelID].setY(y);
              region[voxelID].setZ(z);
              
              voxelID++; 
            }
          }
        }
      }
       int getSize() {
         return pow(size, 3);
       }
       double getDistPair(int vLoc1, int vLoc2) {
         
         assert((vLoc1 < pow(size,3)) && (vLoc2 < pow(size,3)));
         
         VOXEL v1 = region[vLoc1];
         VOXEL v2 = region[vLoc2];
         
         double xDist, yDist, zDist;
         
         xDist = v1.getX() - v2.getX();
         yDist = v1.getY() - v2.getY();
         zDist = v1.getZ() - v2.getZ();
         
         return (pow(xDist,2) + pow(yDist,2) + pow(zDist,2));
       };
};

#endif
