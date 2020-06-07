#include <iostream>
#include <SDL.h>
#include <GL/glew.h>
#include <exception>
#include <chrono>
#include <tuple>
#include <complex>
#include <iomanip>
#include <cassert>
#include <cuda_gl_interop.h>
#include <surface_functions.h>

using namespace std::chrono_literals;

void printVal(GLenum tp, std::string name) {
  const GLubyte* sv = glGetString(tp);
  if (sv == nullptr) {
    std::cerr << "can't get " << name << ": " << glewGetErrorString(glGetError()) << std::endl;
  } else {
    std::cerr << name << ": " << reinterpret_cast<const char*>(sv) << std::endl;
  }
}

class SDLError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SDLOpenGLContext {
  SDL_Window *win = nullptr;
  SDL_GLContext ctx = nullptr;

  SDLOpenGLContext(const std::string& title, int x, int y, int w, int h, uint32_t flags) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
      throw SDLError(std::string("SDL_Init: ") + SDL_GetError());
    }

    assert(0 == SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE));
    assert(0 == SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4));
    assert(0 == SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6));

    win = SDL_CreateWindow(title.c_str(), x, y, w, h, flags);
    if (!win) {
      SDL_Quit();
      throw SDLError(std::string("SDL_CreateWindow: ") + SDL_GetError());
    }

    ctx = SDL_GL_CreateContext(win);
    if (ctx == nullptr) {
      SDL_DestroyWindow(win);
      SDL_Quit();
      throw SDLError(std::string("SDL_GL_CreateContext: ") + SDL_GetError());
    }

    // ////... glewExperimental
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK) {
      SDL_GL_DeleteContext(ctx);
      SDL_DestroyWindow(win);
      SDL_Quit();
      throw std::runtime_error(
          std::string("glewInit: ")
          + reinterpret_cast<const char *>(glewGetErrorString(glewError))
      );
    }

    // maybe vsync
  }

  ~SDLOpenGLContext() {
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
  }
};


//__host__ __device__
float mod(float x, float y) {
  return x - std::floor(x / y) * y;
}

using color = std::tuple<int, int, int>;

//__host__ __device__
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


struct World {
  float pos_x = 0, pos_y = 0;

//  __device__
  color worldAt(float x, float y) {
    x = std::floor(x * 10) / 10;
    y = std::floor(y * 10) / 10;
    return {
        int(mod(int(x * 255), 255)),
        int(mod(int(y * 255), 255)),
        125,
    };
  }

//  __device__
  color viewAt(float dx, float dy, float t) {
    t = std::sin(t);
    t = std::pow(std::abs(t), 1/1.5f) * (t < 0 ? -1 : 1);
    float rad = (dx * dx + dy * dy);
//    auto fin = std::complex<float>(dx, dy) * std::pow(std::complex<float>(std::exp(1)), std::complex<float>(0, std::sqrt(rad) * 5 * t));
//    float fin_dx = fin.real();
//    float fin_dy = fin.imag();

    float phi = std::sqrt(rad) * 5 * t;
    float sin_ = std::sin(phi);
    float cos_ = std::cos(phi);

    float fin_dx = dx * cos_ - dy * sin_;
    float fin_dy = dx * sin_ + dy * cos_;
    int r, g, b;
    auto col = worldAt(pos_x + fin_dx, pos_y + fin_dy);
    r = std::get<0>(col);
    g = std::get<1>(col);
    b = std::get<2>(col);

    float k = std::max(0.3f, 1 - (dx*dx + dy*dy));
    return {r * k, g * k, b * k};
  }
};


struct Axis {
  bool neg = false, pos = false;

  float delta() {
    return (neg * -1.0f + pos * 1.0f);
  }
};

//__global__
//void render(int w, int h, color *colors, World world, float t) {
//  float k = std::min(w, h) / 2.0f;
//
//  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
//  int stride_x = blockDim.x * gridDim.x;
//
//  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
//  int stride_y = blockDim.y * gridDim.y;
//
//  for (int y = idx_y; y < h; y += stride_y) {
//    for (int x = idx_x; x < w; x += stride_x) {
//      float rel_x = (x + 0.5f - w / 2.0f) / k;
//      float rel_y = -(y + 0.5f - h / 2.0f) / k;
//
//      auto &targ = colors[y * w + x];
//      auto col = world.viewAt(rel_x, rel_y, t);
//      std::get<0>(targ) = std::get<0>(col);
//      std::get<1>(targ) = std::get<1>(col);
//      std::get<2>(targ) = std::get<2>(col);
//    }
//  }
//}

template< typename T >
std::string int_to_hex( T i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(sizeof(T)*2)
         << std::hex << i;
  return stream.str();
}

void checkErr(int line_num, std::string line) {
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << line_num << ": " << line << "\ngl error: " << glewGetErrorString(err) << " (" << int_to_hex(err) << ")" << std::endl;
    throw std::runtime_error(std::to_string(err));
  }
}

#define glGuard(expr) \
do { \
  glGetError(); \
  expr; \
  checkErr(__LINE__, #expr); \
} while (false)


void __global__ smt(cudaSurfaceObject_t surf) {
  surf2Dwrite(make_uchar4(255, 0, 0, 0), surf, 0, 0);
};


void dow() {
  const int W = 30;
  const int H = 30;
  auto ctx = SDLOpenGLContext("Hello!", 100, 100, W, H, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);

  printVal(GL_RENDERER, "GL_VENDOR");
  printVal(GL_VENDOR, "GL_RENDERER");
  printVal(GL_VERSION, "GL_VERSION");

  // https://stackoverflow.com/questions/31482816/opengl-is-there-an-easier-way-to-fill-window-with-a-texture-instead-using-vbo
  GLuint fb = 0;
  glGuard(glGenFramebuffers(1, &fb));
  glGuard(glBindFramebuffer(GL_READ_FRAMEBUFFER, fb));

  GLuint tex = 0;
  glGuard(glGenTextures(1, &tex));
  glGuard(glBindTexture(GL_TEXTURE_2D, tex));

  glGuard(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
  glGuard(glBindTexture(GL_TEXTURE_2D, 0));

  glGuard(glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0));

  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

  // https://forums.developer.nvidia.com/t/reading-and-writing-opengl-textures-with-cuda/31746/6
  cudaGraphicsResource *resource;
  assert(cudaSuccess == cudaGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
  assert(cudaSuccess == cudaGraphicsMapResources(1, &resource));

  cudaArray_t writeArray;
  assert(cudaSuccess == cudaGraphicsSubResourceGetMappedArray(&writeArray, resource, 0, 0));

  cudaResourceDesc descr = {};
  descr.resType = cudaResourceTypeArray;
  descr.res.array.array = writeArray;

  cudaSurfaceObject_t surf;
  assert(cudaSuccess == cudaCreateSurfaceObject(&surf, &descr));

  smt<<<1, 1>>>(surf);
  cudaDeviceSynchronize();

//  gpu::error::check(cudaGraphicsUnmapResources(1, &writeresource, 0));
//
//  gpu::error::check(cudaGraphicsUnregisterResource(writeresource));
//  int dbl = 123;
//  glGuard(glGetIntegerv(GL_DOUBLEBUFFER, &dbl));
//  std::cout << "dbl: " << dbl << std::endl;

  Axis dx, dy;
  World world;

//  color *colors;
//  cudaMallocManaged(&colors, sizeof(color) * W * H);

  auto prev_frame = std::chrono::steady_clock::now();
  auto start = prev_frame;
  while (true) {
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration_cast<std::chrono::duration<float, std::chrono::seconds::period>>(now - prev_frame).count();
    prev_frame = now;

    std::cout << dt - 1/60.0f << std::endl;

    SDL_Event evt;
    bool quit = false;
    while (SDL_PollEvent(&evt)) {
      if (evt.type == SDL_QUIT) {
        quit = true;
      }
      if (evt.type == SDL_KEYDOWN) {
        switch (evt.key.keysym.sym) {
          case (SDLK_d):
            dx.pos = true; break;
          case (SDLK_a):
            dx.neg = true; break;
          case (SDLK_w):
            dy.pos = true; break;
          case (SDLK_s):
            dy.neg = true; break;
        }
      }
      if (evt.type == SDL_KEYUP) {
        switch (evt.key.keysym.sym) {
          case (SDLK_d):
            dx.pos = false; break;
          case (SDLK_a):
            dx.neg = false; break;
          case (SDLK_w):
            dy.pos = false; break;
          case (SDLK_s):
            dy.neg = false; break;
        }
      }
    }
    if (quit)
      break;

    float t = std::chrono::duration_cast<std::chrono::duration<float, std::chrono::seconds::period>>(prev_frame - start).count();
    float a = std::sin(t);
    float b = std::sin(t);

    world.pos_x += a * dt;
    world.pos_y += b * dt;

    //    render<<<10, 256>>>(surf->w, surf->h, colors, world, t);
    //    cudaDeviceSynchronize();

    // TODO: clear?
    glGuard(glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST));

    SDL_GL_SwapWindow(ctx.win);

    auto left = std::chrono::steady_clock::now() - prev_frame;
    SDL_Delay(std::max(0.0f, 1000.0f / 60 - std::chrono::duration_cast<std::chrono::milliseconds>(left).count()));
  }
//  cudaFree(colors);
}

int main(int, char**) {
  try {
    dow();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}