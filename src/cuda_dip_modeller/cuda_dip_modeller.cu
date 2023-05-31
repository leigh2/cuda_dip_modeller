#include <float.h>

// fill the stellar grid array
__global__ void make_mugrid(
    const int n_elem_per_dim,
    const int overbin,
    double * xgrid,
    double * ygrid,
    double * mugrid
){
    // thread location, return if out of bounds
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_idx >= n_elem_per_dim || y_idx >= n_elem_per_dim) return;

    // 2d array pointer
    const int ptr2d = x_idx + n_elem_per_dim*y_idx;

    // populate xgrid and ygrid arrays
    double xloc = xgrid[ptr2d] = 2.0 * (x_idx + 0.5) / n_elem_per_dim - 1.0;
    double yloc = ygrid[ptr2d] = 2.0 * (y_idx + 0.5) / n_elem_per_dim - 1.0;

    // distance from center of the star squared
    double d2 = xloc * xloc + yloc * yloc;

    // populate the mu array
    mugrid[ptr2d] = sqrtf(1 - d2);

}

// calculate the flux at each grid-time point pair
__global__ void populate_flux_points(
    const double * xc,
    const double * yc,
    const int n_pos_elem,
    const double transp,
    const double r0,
    const double rmaj,
    const double rmin,
    const double pa,
    const double alimb,
    const double blimb,
    const double * xgrid,
    const double * ygrid,
    const double * mugrid,
    const int n_grid_elem,
    double * lc
){
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_idx >= n_pos_elem || y_idx >= n_grid_elem) return;
}