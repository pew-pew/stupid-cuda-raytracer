#pragma once
#include <cmath>
#include "geom.h"

__host__ __device__
Vec twobody(float mu, float tau, Vec ri, Vec vi) {
  /*
  % solve the two body initial value problem
  % Goodyear's method

  % input
  %  mu  = gravitational constant (km**3/sec**2)
  %  tau = propagation time interval (seconds)
  %  ri  = initial eci position vector (kilometers)
  %  vi  = initial eci velocity vector (km/sec)
  % output
  %  rf = final eci position vector (kilometers)
  %  //////////vf = final eci velocity vector (km/sec)
  % Orbital Mechanics with MATLAB
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  */
  float a0 = 0.025;
  float b0 = a0 / 42;
  float c0 = b0 / 72;
  float d0 = c0 / 110;
  float e0 = d0 / 156;
  float f0 = e0 / 210;
  float g0 = f0 / 272;
  float h0 = g0 / 342;
  float i0 = 1.0f / 24;
  float j0 = i0 / 30;
  float k0 = j0 / 56;
  float l0 = k0 / 90;
  float m0 = l0 / 132;
  float n0 = m0 / 182;
  float o0 = n0 / 240;
  float p0 = o0 / 306;
  //% convergence criterion
  float tol = 1.0e-8;
  float rsdvs = ri.dot(vi);
  float rsm = ri.len();
  float vsm2 = vi.dot(vi);
  float zsma = 2.0f / rsm - vsm2 / mu;

  float psi;

  if (zsma > 0.0) {
    psi = tau * zsma;
  } else {
    psi = 0.0;
  }

  float rfm, s1, s2, gg;
  float alp = vsm2 - 2.0f * mu / rsm;
  for (int z = 1; z <= 20; z++) {
    float m = 0;
    float psi2 = psi * psi;
    float psi3 = psi * psi2;
    float aas = alp * psi2;
    float zas;
    if (std::abs(aas) < 1e-9) {
      zas = 1 / aas;
    } else {
      zas = 0;
    }
    while (std::abs(aas) > 1) {
      m = m + 1;
      aas = 0.25 * aas;
    }
    float pc5 = a0 + (b0 + (c0 + (d0 + (e0 + (f0 + (g0 + h0 * aas) * aas) * aas) * aas) * aas) * aas) * aas;
    float pc4 = i0 + (j0 + (k0 + (l0 + (m0 + (n0 + (o0 + p0 * aas) * aas) * aas) * aas) * aas) * aas) * aas;
    float pc3 = (0.5 + aas * pc5) / 3;
    float pc2 = 0.5 + aas * pc4;
    float pc1 = 1.0 + aas * pc3;
    float pc0 = 1.0 + aas * pc2;
    if (m > 0) {
      while (m > 0) {
        m = m - 1;
        pc1 = pc0 * pc1;
        pc0 = 2 * pc0 * pc0 - 1;
      }
      pc2 = (pc0 - 1) * zas;
      pc3 = (pc1 - 1) * zas;
    }
    s1 = pc1 * psi;
    s2 = pc2 * psi2;
    float s3 = pc3 * psi3;
    gg = rsm * s1 + rsdvs * s2;
    float dtau = gg + mu * s3 - tau;
    rfm = std::abs(rsdvs * s1 + mu * s2 + rsm * pc0);
    if (std::abs(dtau) < std::abs(tau) * tol) {
      break;
    } else {
      psi = psi - dtau / rfm;
    }
  }
  float rsc = 1 / rsm;
  float r2 = 1 / rfm;
  float r12 = rsc * r2;
  float fm1 = -mu * s2 * rsc;
  float ff = fm1 + 1;
  float fd = -mu * s1 * r12;
  float gdm1 = -mu * s2 * r2;
  float gd = gdm1 + 1;
  // % compute final state vector
  Vec rf = ff * ri + gg * vi;
  Vec vf = fd * ri + gd * vi;
  return rf;
}
