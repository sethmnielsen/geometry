#pragma once

#include <vector>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;


template <typename T>
class Camera
{

enum {FX, FY, CX, CY, RX, RY, D1, D2, D3, D4, D5, S, BUF_SIZE};

public:
    typedef Matrix<T,2,1> Vec2;
    typedef Matrix<T,2,2> Mat2;
    typedef Matrix<T,3,1> Vec3;
    typedef Matrix<T,5,1> Vec5;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Camera() :
        focal_len_(buf_),
        cam_center_(buf_+CX),
        image_size_(buf_+RX),
        distortion_(buf_+D1),
        s_(buf_[S])
    {
      focal_len_.setZero();
      cam_center_.setZero();
      image_size_.setZero();
      distortion_.setZero();
      s_ = T(0);
    }

    Camera(const Vec2& f, const Vec2& c, const Vec5& d, const T& s) :
        focal_len_(const_cast<T*>(f.data())),
        cam_center_(const_cast<T*>(c.data())),
        image_size_(buf_+RX),
        distortion_(const_cast<T*>(d.data())),
        s_(*const_cast<T*>(&s))
    {
      K_(0,0) = f(0);
      K_(1,1) = f(1);
      K_(0,1) = s;
      K_(0,2) = c(0);
      K_(1,2) = c(1);
    }


    Camera(const T* f, const T* c, const T* d, const T* s) :
        focal_len_ (const_cast<T*>(f)),
        cam_center_(const_cast<T*>(c)),
        image_size_(buf_+RX),
        distortion_(const_cast<T*>(d)),
        s_(*const_cast<T*>(s))
    {
      K_(0,0) = *f;
      K_(1,1) = *(f+1);
      K_(0,1) = *s;
      K_(0,2) = *c;
      K_(1,2) = *(c+1);
    }

    Camera(const Vec2& f, const Vec2& c, const Vec5& d, const T& s, const Vec2& size) :
        focal_len_(const_cast<T*>(f.data())),
        cam_center_(const_cast<T*>(c.data())),
        image_size_(const_cast<T*>(size.data())),
        distortion_(const_cast<T*>(d.data())),
        s_(*const_cast<T*>(&s))
    {
      K_(0,0) = f(0);
      K_(1,1) = f(1);
      K_(0,1) = s;
      K_(0,2) = c(0);
      K_(1,2) = c(1);
    }

    Camera(const T* f, const T* c, const T* d, const T* s, const T* size) :
        focal_len_ (const_cast<T*>(f)),
        cam_center_(const_cast<T*>(c)),
        image_size_(const_cast<T*>(size)),
        distortion_(const_cast<T*>(d)),
        s_(*const_cast<T*>(s))
    {
      K_(0,0) = *f;
      K_(1,1) = *(f+1);
      K_(0,1) = *s;
      K_(0,2) = *c;
      K_(1,2) = *(c+1);
    }

    Camera& operator=(const Camera& cam)
    {
      focal_len_ = cam.focal_len_;
      cam_center_ = cam.cam_center_;
      distortion_ = cam.distortion_;
      s_ = cam.s_;
      image_size_ = cam.image_size_;
      K_ = cam.K_;
    }

    template<typename T2>
    const Camera<T2> cast() const
    {
      Camera<T2> cam;
      cam.focal_len_ = focal_len_.template cast<T2>();
      cam.cam_center_ = cam_center_.template cast<T2>();
      cam.image_size_ = image_size_.template cast<T2>();
      cam.distortion_ = distortion_.template cast<T2>();
      cam.s_ = T2(s_);
      cam.K_ = K_.template cast<T2>();
      return cam;
    }

    void unDistort(const Vec2& pi_u, Vec2& pi_d) const
    {
        const T k1 = distortion_(0);
        const T k2 = distortion_(1);
        const T p1 = distortion_(2);
        const T p2 = distortion_(3);
        const T k3 = distortion_(4);
        const T x = pi_u.x();
        const T y = pi_u.y();
        const T xy = x*y;
        const T xx = x*x;
        const T yy = y*y;
        const T rr = xx*yy;
        const T r4 = rr*rr;
        const T r6 = r4*rr;


        // https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        const T g =  1.0 + k1 * rr + k2 * r4 + k3*r6;
        const T dx = 2.0 * p1 * xy + p2 * (rr + 2.0 * xx);
        const T dy = 2.0 * p2 * xy + p1 * (rr + 2.0 * yy);

        pi_d.x() = g * (x + dx);
        pi_d.y() = g * (y + dy);
    }

    void Distort(const Vec2& pi_d, Vec2& pi_u, double tol=1e-6) const
    {
        pi_u = pi_d;

        if (distortion_(0) == 0)
            return;

        Vec2 pihat_d;
        Mat2 J;
        Vec2 e;
        T prev_e = (T)1000.0;
        T enorm = (T)0.0;

        static const int max_iter = 50;
        int i = 0;
        while (i < max_iter)
        {
            unDistort(pi_u, pihat_d);
            e = pihat_d - pi_d;
            enorm = e.norm();
            if (enorm <= tol || prev_e < enorm)
                break;
            prev_e = enorm;

            distortJac(pi_u, J);
            pi_u = pi_u - J*e;
            i++;
        }
    }

    void distortJac(const Vec2& pi_u, Mat2& J) const
    {
        const T k1 = distortion_(0);
        const T k2 = distortion_(1);
        const T p1 = distortion_(2);
        const T p2 = distortion_(3);
        const T k3 = distortion_(4);

        const T x = pi_u.x();
        const T y = pi_u.y();
        const T xy = x*y;
        const T xx = x*x;
        const T yy = y*y;
        const T rr = xx+yy;
        const T r = sqrt(rr);
        const T r4 = rr*rr;
        const T r6 = rr*r4;
        const T g =  (T)1.0 + k1 * rr + k2 * r4 + k3*r6;
        const T dx = (x + ((T)2.0*p1*xy + p2*(rr+(T)2.0*xx)));
        const T dy = (y + (p1*(rr+(T)2.0*yy) + (T)2.0*p2*xy));

        const T drdx = x / r;
        const T drdy = y / r;
        const T dgdx = k1*(T)2.0*r*drdx + (T)4.0*k2*rr*r*drdx + (T)6.0*k3*r4*r*drdx;
        const T dgdy = k1*(T)2.0*r*drdy + (T)4.0*k2*rr*r*drdy + (T)6.0*k3*r4*r*drdy;

        J << /* dxbar/dx */ ((T)1.0 + ((T)2.0*p1*y + p2*((T)2.0*r*drdx + (T)4.0*x)))*g + dx*dgdx,
             /* dxbar/dy */ ((T)2.0*p1*x + p2*(T)2.0*r*drdy)*g + dx*dgdy,
             /* dybar/dx */ (p1*(T)2.0*r*drdx+(T)2.0*p2*y)*g + dy*dgdx,
             /* dybar/dy */ ((T)1.0 + (p1*((T)2.0*r*drdy + (T)4.0*y) + (T)2.0*p2*x))*g + dy*dgdy;

        if ((J.array() != J.array()).any())
        {
            int debug = 1;
        }
    }


    void pix2intrinsic(const Vec2& pix, Vec2& pi) const
    {
        const T fx = focal_len_.x();
        const T fy = focal_len_.y();
        const T cx = cam_center_.x();
        const T cy = cam_center_.y();
        pi << (1.0/fx) * (pix.x() - cx - (s_/fy) * (pix.y() - cy)),
                (1.0/fy) * (pix.y() - cy);
    }

    void intrinsic2pix(const Vec2& pi, Vec2& pix) const
    {
        const T fx = focal_len_.x();
        const T fy = focal_len_.y();
        const T cx = cam_center_.x();
        const T cy = cam_center_.y();
        pix << fx*pi.x() + s_*pi.y() + cx,
               fy*pi.y() + cy;
    }

    void proj(const Vec3& pt, Vec2& pix) const
    {
        const T pt_z = pt(2);
        Vec2 pi_d;
        Vec2 pi_u = (pt.template segment<2>(0) / pt_z);
        Distort(pi_u, pi_d);
        intrinsic2pix(pi_d, pix);
    }

    Vec2 proj(const Vec3& pt) const
    {
        Vec2 pix;
        proj(pt, pix);
        return pix;
    }

    inline bool check(const Vector2d& pix) const
    {
        return !((pix.array() > image_size_.array()).any()|| (pix.array() < 0).any());
    }

    void invProj(const Vec2& pix, const T& depth, Vec3& pt) const
    {
        Vec2 pi_d, pi_u;
        pix2intrinsic(pix, pi_d);
        unDistort(pi_d, pi_u);
        pt.template segment<2>(0) = pi_u;
        pt(2) = (T)1.0;
        pt *= depth / pt.norm();
    }

    Vec3 invProj(const Vec2& pix, const T& depth) const
    {
        Vec3 zeta;
        invProj(pix, depth, zeta);
        return zeta;
    }

    Map<Vec2> focal_len_;
    Map<Vec2> cam_center_;
    Map<Vec2> image_size_;
    Map<Vec5> distortion_;
    T& s_;
    Matrix<T,3,3> K_ = Matrix<T,3,3>::Identity(); // intrinsic camera matrix

private:
    const Matrix2d I_2x2 = Matrix2d::Identity();
    T buf_[BUF_SIZE]; // [fx, fy, cx, cy, size_x, size_y, d1, d2, d3, d4, d5, s]
};

typedef Camera<double> Camerad;
