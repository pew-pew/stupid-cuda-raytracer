struct Vec {
  float x, y;

  __host__ __device__
  Vec(): x(0), y(0) {}

  __host__ __device__
  Vec(float x, float y): x(x), y(y) {}

  __host__ __device__
  friend Vec operator+(const Vec& a, const Vec& b) {
    return Vec(a.x + b.x, a.y + b.y);
  }

  __host__ __device__
  friend Vec operator-(const Vec& a, const Vec& b) {
    return Vec(a.x - b.x, a.y - b.y);
  }

  __host__ __device__
  friend Vec operator-(const Vec& v) {
    return Vec(-v.x, -v.y);
  }

  __host__ __device__
  friend Vec operator*(const Vec& v, float k) {
    return Vec(v.x * k, v.y * k);
  }

  __host__ __device__
  friend Vec operator/(const Vec& v, float k) {
    return Vec(v.x / k, v.y / k);
  }

  __host__ __device__
  float dot(const Vec& u) const {
    return x * u.x + y * u.y;
  }

  __host__ __device__
  float cross(const Vec& u) const {
    return x * u.y - y * u.x;
  }

  __host__ __device__
  float len() const {
    return std::sqrt(x * x + y * y);
  }

  __host__ __device__
  float lensq() const {
    return x * x + y * y;
  }

  __host__ __device__
  Vec symmetryOff(Vec ax) const {
    float a = (ax.x * ax.x - ax.y * ax.y) / ax.lensq();
    float b = (2 * ax.x * ax.y) / ax.lensq();
    return Vec(
      x * a + y * b,
      x * b - y * a
    );
  }

  __host__ __device__
  Vec rotateBy(float phi) const {
    float s = std::sin(phi);
    float c = std::cos(phi);
    return Vec(
      x * c - y * s,
      x * s + y * c
    );
  }
};


__host__ __device__
float intersectionTime(Vec a0, Vec da, Vec b0, Vec db) {
  return -(a0 - b0).cross(db) / da.cross(db);
}