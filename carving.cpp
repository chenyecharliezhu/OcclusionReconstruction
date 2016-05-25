#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <cmath>

using namespace std;

struct Voxel {
  float x, y, z;
  float depth;
  unsigned char r, g, b;
  unsigned char camera_index;
};

void voxel_carving(string dir, int total_voxels);

int main(int argc, char** argv) {
  string dir = string(argv[1]);
  int total_voxels = argc > 2 ? atoi(argv[2]) : 6000000;

  voxel_carving(dir, total_voxels);
  return 0;
}

void space_carve(const cv::Mat& projection, const cv::Mat& mask,
    const cv::Mat& img, vector<Voxel>& voxels, int indx) {

  cv::Mat X(3, 1, CV_32F);
  cv::Mat X2(4, 1, CV_32F);
  vector<Voxel> visible_voxels;

  /*
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++)
      cout << projection.at<float>(i,j) << " ";
    cout << endl;
  }
  */
    
  for(size_t i = 0; i < voxels.size(); i++) {
    Voxel &vox = voxels[i];

    X2.at<float>(0) = vox.x;
    X2.at<float>(1) = vox.y;
    X2.at<float>(2) = vox.z;
    X2.at<float>(3) = 1.0;

    X = projection * X2;
    int x = X.at<float>(0) / X.at<float>(2);
    int y = X.at<float>(1) / X.at<float>(2);

    if(x < 0 || x >= mask.cols || y < 0 || y >= mask.rows) {
      continue;
    }

    if(mask.at<uchar>(y,x)) {
      if(X.at<float>(2) < vox.depth) {
        vox.depth = X.at<float>(2);
        vox.r = img.at<cv::Vec3b>(y,x)[2];
        vox.g = img.at<cv::Vec3b>(y,x)[1];
        vox.b = img.at<cv::Vec3b>(y,x)[0];
        vox.camera_index = indx;
      }
      visible_voxels.push_back(vox);
    }
  }
  voxels = visible_voxels;
}

void generate_silhouette(cv::Mat& mask, const cv::Mat& img) {
  for (int y = 0; y < mask.rows; y++) {
    for(int x = 0; x < mask.cols; x++) {
      int r = img.at<cv::Vec3b>(y,x)[2];
      int g = img.at<cv::Vec3b>(y,x)[1];
      int b = img.at<cv::Vec3b>(y,x)[0];

      if ((g < r || g < b) && (r + g + b > 200)) {
      //if (r < g || r < b) {
        mask.at<uchar>(y,x) = 0;
      }
    }
  }
}

void voxel_carving(string dir, int total_voxels) {
  char line[1024];
  int num;
  vector<cv::Mat> imgs;
  vector<cv::Mat> projections;
  vector<cv::Mat> masks;
  vector<Voxel> voxel_grid;

  double x_low = -1;
  double x_high = 1;
  double y_low = -1;
  double y_high = 2;
  double z_low = 0;
  double z_high = 0.1;

  double total_volume = (x_high - x_low) * (y_high - y_low) * (z_high - z_low);

  double voxel_size = pow(total_volume / total_voxels, 1.0 / 3); 
  int xdiv = (int)(round((x_high - x_low) / voxel_size));
  int ydiv = (int)(round((y_high - y_low) / voxel_size));
  int zdiv = (int)(round((z_high - z_low) / voxel_size));

  // Init grid
  voxel_grid.resize(xdiv * ydiv * zdiv);

  int k = 0;
  for (int x = 0; x < xdiv; x++) {
    for (int y = 0; y < ydiv; y++) {
      for (int z = 0; z < zdiv; z++) {
        voxel_grid[k].x = x_low + (x + 0.5) * voxel_size;
        voxel_grid[k].y = y_low + (y + 0.5) * voxel_size;
        voxel_grid[k].z = z_low + (z + 0.5) * voxel_size;
        voxel_grid[k].depth = FLT_MAX;

        k++;
      }
    }
  }

  // Read in projection matrices
  ifstream input(dir + "/P.txt");
  assert(input);

  input.getline(line, sizeof(line));
  sscanf(line, "%d", &num);

  for (int i = 0; i < num; i++) {
    cv::Mat P(3, 4, CV_32F);
    input.getline(line, sizeof(line)); // blank

    for (int j = 0; j < 3; j++) {
      input.getline(line, sizeof(line));
      stringstream a(line);

      a >> P.at<float>(j, 0);
      a >> P.at<float>(j, 1);
      a >> P.at<float>(j, 2);
      a >> P.at<float>(j, 3);
    }

    projections.push_back(P);
  }


  for(size_t k = 0; k < projections.size(); k++) {
    char str[128];
    cv::Mat img, mask;

    sprintf(str, "%02d.jpg", (int)k);
    cout << "Loading " <<  str << endl;

    img = cv::imread(dir + '/' + str);
    assert(img.data);

    mask = cv::Mat::ones(img.size(), CV_8U) * 255;

    generate_silhouette(mask, img);
    // Fill up any holes
    dilate(mask, mask, cv::Mat());

    imgs.push_back(img);
    masks.push_back(mask);

    // Visualization
    {
      cv::Mat img2, mask2;

      cv::resize(img, img2, cv::Size(img.cols/2, img.rows/2));
      cv::resize(mask, mask2, cv::Size(img.cols/2, img.rows/2));
      cv::cvtColor(mask2, mask2, cv::COLOR_GRAY2BGR);

      cv::Mat canvas(img2.rows, img2.cols*2, CV_8UC3);

      img2.copyTo(canvas(cv::Rect(0, 0, img2.cols, img2.rows)));
      mask2.copyTo(canvas(cv::Rect(img2.cols, 0, img2.cols, img2.rows)));

      cv::imshow("main", canvas);

      cv::waitKey(200);
    }

    //if (k == 8 || k == 7 || k == 9) {
    if (k < 2) {
      space_carve(projections[k], mask, img, voxel_grid, k);
    }
  }

  // Output to Meshlab readable file
  cout << "Saving to output.ply" << endl;

  ofstream output("output.ply");

  output << "ply" << endl;
  output << "format ascii 1.0" << endl;
  output << "element vertex " << voxel_grid.size() << endl;
  output << "property float x" << endl;
  output << "property float y" << endl;
  output << "property float z" << endl;
  output << "property uchar diffuse_red" << endl;
  output << "property uchar diffuse_green" << endl;
  output << "property uchar diffuse_blue" << endl;
  output << "end_header" << endl;

  for(size_t i=0; i < voxel_grid.size(); i++) {
    Voxel &v = voxel_grid[i];
    output << v.x << " " << v.y << " " << v.z << " " << (int)v.r << " " << (int)v.g << " " << (int)v.b << endl;
  }

  cout << voxel_grid.size() << endl;
}
