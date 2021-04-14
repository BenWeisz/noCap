import java.util.*;

// PLEASE DO THE FOLLOWING TO RUN THE CODE:
// Place the following files in this folder structure (name them exactly as such):
// Unfortunately Markus does not allow me to upload directories.
//
// model_maker (dir)
//  |
//  >> model_maker.pde
//  |
//  >> data (dir)
//      |
//      >> bkg.png

class Pair {
  public int x;
  public int y;
  
  public Pair(int x, int y) {
    this.x = x;
    this.y = y;
  }
}

PrintWriter output;
PImage bkg;

LinkedList<Pair> verts;
LinkedList<Pair> edges;

boolean isEditingVerts = true;
int first;

void setup() {
  size(640, 480);
  noSmooth();
  
  verts = new LinkedList<Pair>();
  edges = new LinkedList<Pair>();
  first = -1;
  
  output = createWriter("model.obj");
  bkg = loadImage("bkg.png");
  
  println("2D Mesh Model Builder");
  println("=====================");
  println("This program is used to build the mesh model used by the motion capture app.");
  println("To run this program you need to install Processing from: https://processing.org/download/");
  println("");
  println("Usage:");
  println("When in vertex mode, clicking adds new verticies.");
  println("When in edge mode, first click near one vertex and then another to connect them.");
  println("Press e or E to toggle between editing verticies and edges");
  println("");
  println("If you mess up an edge or vertex placement, press r or R to remove the last entry");
  println("");
  println("When your model is ready, press f to export the file into the data folder.");
  println("If you wish to change the output file name, please rename the output file in this function.");
}

void draw() {
  background(255, 255, 255);
  image(bkg, 0, 0);
  
  if (isEditingVerts) {
    fill(0, 0, 0);
    text("Verts", 10, 10);
    fill(255, 255, 255);
  }
  else {
    fill(0, 0, 0);
    text("Edges", 10, 10);
    fill(100, 100, 100);
  }
  
  for (int i = 0; i < verts.size(); i++) {
    Pair p = verts.get(i);
    if (i == first) {
      fill(255, 0, 0);
    }
    else {
      if (isEditingVerts) {
        fill(255, 255, 255);
      }
      else {
        fill(100, 100, 100);
      }
    }
    circle(p.x, p.y, 10);
  }
   
  for (Pair p : edges) {
    Pair s1 = verts.get(p.x);
    Pair s2 = verts.get(p.y);
    
    line(s1.x, s1.y, s2.x, s2.y);
  }
}

boolean canAddPair(int first, int second) {
  if (first == second)
    return false;
  
  for (Pair p : edges) {
    if ((p.x == first && p.y == second) || (p.x == second && p.y == first)) {
      return false;
    }
  }
  
  return true;
}

void mouseClicked() {
  if (isEditingVerts) {
    Pair p = new Pair(mouseX, mouseY);
    verts.add(p);
  }
  else {
    int closestInd = 0;
    double closestDist = 100000000;
    
    for (int i = 0; i < verts.size(); i++) {
      Pair p = verts.get(i);
      double dist = Math.sqrt(((p.x - mouseX) * (p.x - mouseX)) + ((p.y - mouseY) * (p.y - mouseY)));
      if (dist < closestDist) {
        closestDist = dist;
        closestInd = i;
      }
    }
    
    if (first == -1) {
      first = closestInd;
    }
    else {
      Pair p = new Pair(first, closestInd);
      if (canAddPair(first, closestInd))
        edges.add(p);
      first = -1;
    }
  }
}

void export() {
  for (Pair p : verts) {
    output.println("V " + p.x + " " + p.y);
  }
  
  for (Pair p : edges) {
    output.println("E " + p.x + " " + p.y);
  }
  
  output.flush();
  output.close();
  exit();
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    if (isEditingVerts) {
      if (verts.size() > 0)
        verts.removeLast();
    }
    else {
      if (edges.size() > 0)
        edges.removeLast();
    }
  }
  else if (key == 'e' || key == 'E') {
    isEditingVerts = false;
  }
  else if (key == 'f' || key == 'F') {
    export();
  }
  else {
    isEditingVerts = true;
  }
}
