#pragma once
#include <cmath>

__host__ __device__
float mod(float x, float y) {
  return x - std::floor(x / y) * y;
}

using color = std::tuple<int, int, int>;

__host__ __device__
color hsl2rgb(float h, float s, float l) {
  h = mod(h, 360);
  float c = (1 - std::abs(2 * l - 1)) * s;
  float x = c * (1 - std::abs(mod(h / 60, 2) - 1));
  float m = l - c / 2;
  float r_ = 0, g_ = 0, b_ = 0;
  if (h < 60) {
    r_ = c; g_ = x;
  } else if (h < 120) {
    r_ = x; g_ = c;
  } else if (h < 180) {
    g_ = c; b_ = x;
  } else if (h < 240) {
    g_ = x; b_ = c;
  } else if (h < 300) {
    r_ = x; b_ = c;
  } else {
    r_ = c; b_ = x;
  }
  return {(r_ + m) * 255, (g_ + m) * 255, (b_ + m) * 255};
}


struct RayState {
  Vec pos, dir;
  int steps;
};


struct World {
  Vec curr_pos = Vec(0, 0);
  Vec curr_dir = Vec(0, 1);

  __device__
  color worldAt(Vec pos) {
//    if ((pos - curr_pos).lensq() < 0.1 * 0.1) {
//      return {0, 60, 20};
//    }

    float x = std::floor(pos.x * 10) / 10;
    float y = std::floor(pos.y * 10) / 10;
    return {
        int(mod(int(x * 255), 255)),
        int(mod(int(y * 255), 255)),
        (int(std::floor(x) + std::floor(y)) % 2) * 120,
    };
  }

  __host__ __device__
  RayState trace(Vec pos, Vec dir, float d, float t) {
    Vec m0(1, 0);
    Vec d0 = Vec(1, 0);//.rotateBy(t / 10);

    Vec m1(1, 0.7);
    Vec d1 = Vec(2, 0);//.rotateBy(t);
    m1 -= d1 / 2;

    int refl = 0;

    int lastInters = -1;
    while (refl < 10) {
      float tMy0 = intersectionTime(pos, dir * d, m0, d0);
      float tOther0 = intersectionTime(m0, d0, pos, dir * d);

      float tMy1 = intersectionTime(pos, dir * d, m1, d1);
      float tOther1 = intersectionTime(m1, d1, pos, dir * d);

      bool inters0 = (0 <= tOther0 && tOther0 <= 1 && 0 <= tMy0 && tMy0 <= 1 && lastInters != 0);
      bool inters1 = (0 <= tOther1 && tOther1 <= 1 && 0 <= tMy1 && tMy1 <= 1 && lastInters != 1);

      if (inters0 && (!inters1 || tMy0 < tMy1)) {
//        if ((pos - m0).isLeftTo(d0)) {
//          refl = 100;
////          pos = pos + delta * (tMy0 * 0.99);
//          break;
//        }

        refl++;
        Vec mid = m1 + d1 * tOther0;
        pos = mid;
        dir = dir.ortoRotate(d0, d1);
        d *= (1 - tMy0);
        lastInters = 1;
      } else if (inters1 && (!inters0 || tMy1 < tMy0)) {
//        if ((pos - m1).isLeftTo(-d1)) {
//          refl = 100;
////          pos = pos + delta * (tMy1 * 0.99);
//          break;
//        }

        refl++;
        Vec mid = m0 + d0 * tOther1;
        pos = mid;
        dir = dir.ortoRotate(d1, d0);
        d *= (1 - tMy1);
        lastInters = 0;
      } else {
        pos += dir * d;
        break;
      }
    }

    return {pos, dir, refl};
  }

  __device__
  color viewAt(float dx, float dy, float t) {
    if (dx*dx + dy*dy < 0.1 * 0.1) {
      return {0, 60, 20};
    }

    float brightness = 1;

    Vec d = Vec(dx, dy).ortoRotate(Vec(0, 1), curr_dir);

    auto res = trace(curr_pos, d, 1, t);
    auto fin = res.pos;
    auto refl = res.steps;
    brightness /= (1 + refl);

    int r, g, b;
    auto col = worldAt(fin);
    r = std::get<0>(col);
    g = std::get<1>(col);
    b = std::get<2>(col);

    if (brightness < 50) {
      brightness = 1;
    }

//    float k = std::max(0.3f, 1 - (dx*dx + dy*dy));
    return {r * brightness, g * brightness, b * brightness};
  }

  __host__ __device__
  void walk(Vec dir, float d, float t) {
    if (dir == Vec{0, 0})
      return;

    dir = dir.ortoRotate(Vec(0, 1), curr_dir);

    auto newView = trace(
        curr_pos,
        dir,
        d, t);
    curr_pos = newView.pos;
    curr_dir = curr_dir.ortoRotate(dir, newView.dir);
  }

  __host__ __device__
  void zoom(float k) {
    curr_dir *= std::exp(k);
  }
};


