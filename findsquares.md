
```cpp
// find contours and store them all as a list
findContours(gray, contours, ...);

vector<cv::Point> approx;

// test each contour
for (size_t i = 0; i < contours.size(); i++)
{
  // approximate contour with accuracy proportional to the contour perimeter
	approxPolyDP(cv::Mat(contours[i]), approx, ...);

  if (/* check that contour has 4 sides, angles are 90°, etc.*/)
  {
    squares.push_back(approx);
  }
}
```
