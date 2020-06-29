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
#include "geom.h"
#include "world.h"

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


struct Axis {
  bool neg = false, pos = false;

  float delta() {
    return (neg * -1.0f + pos * 1.0f);
  }
};

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



__global__
void render(int w, int h, cudaSurfaceObject_t surf, World world, float t) {
  float k = std::min(w, h) / 2.0f;

  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = blockDim.x * gridDim.x;

  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride_y = blockDim.y * gridDim.y;

  for (int y = idx_y; y < h; y += stride_y) {
    for (int x = idx_x; x < w; x += stride_x) {
      float rel_x = (x + 0.5f - w / 2.0f) / k;
      float rel_y = (y + 0.5f - h / 2.0f) / k;

      auto col = world.viewAt(rel_x, rel_y, t);
      surf2Dwrite(make_uchar4(
          std::get<0>(col),
          std::get<1>(col),
          std::get<2>(col),
          0
      ), surf, x * 4, y);
    }
  }
}


void dow() {
  const int W = 1900;
  const int H = 1000;
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

  Axis dx, dy, dphi, dz;
  World world;

  std::cout << std::endl;

  auto prev_frame = std::chrono::steady_clock::now();
  auto start = prev_frame;
  while (true) {
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration_cast<std::chrono::duration<float, std::chrono::seconds::period>>(now - prev_frame).count();
    prev_frame = now;

    std::cout << world.curr_pos.x << " " << world.curr_pos.y << "      "
              << world.curr_dir.x << " " << world.curr_dir.y << "      "
              << dt - 1/60.0f << "                            \r";

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
          case (SDLK_q):
            dphi.pos = true; break;
          case (SDLK_e):
            dphi.neg = true; break;
          case (SDLK_z):
            dz.pos = true; break;
          case (SDLK_x):
            dz.neg = true; break;
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
          case (SDLK_q):
            dphi.pos = false; break;
          case (SDLK_e):
            dphi.neg = false; break;
          case (SDLK_z):
            dz.pos = false; break;
          case (SDLK_x):
            dz.neg = false; break;
        }
      }
    }
    if (quit)
      break;

    float t = std::chrono::duration_cast<std::chrono::duration<float, std::chrono::seconds::period>>(prev_frame - start).count();

    world.curr_dir = world.curr_dir.rotateBy(dphi.delta() * dt);

    Vec relWalkDir = Vec(
        dx.delta(),
        dy.delta()
    );
    world.walk(relWalkDir, dt, t);
    world.zoom(dz.delta() * dt);

    render<<<10, 256>>>(W, H, surf, world, t);
    cudaDeviceSynchronize();

    // TODO: clear?
    glGuard(glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST));

    SDL_GL_SwapWindow(ctx.win);

    auto left = std::chrono::steady_clock::now() - prev_frame;
    SDL_Delay(std::max(0.0f, 1000.0f / 60 - std::chrono::duration_cast<std::chrono::milliseconds>(left).count()));
  }

  assert(cudaSuccess == cudaDestroySurfaceObject(surf));
  assert(cudaSuccess == cudaGraphicsUnmapResources(1, &resource));
  assert(cudaSuccess == cudaGraphicsUnregisterResource(resource));
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