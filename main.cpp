#include <iostream>
#include <SDL.h>
#include <GL/glew.h>
#include <exception>
#include <chrono>
#include <tuple>
#include <complex>

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
  SDL_Renderer *renderer = nullptr;
  SDL_GLContext ctx = nullptr;

  SDLOpenGLContext(const std::string& title, int x, int y, int w, int h, uint32_t flags) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
      throw SDLError(std::string("SDL_Init: ") + SDL_GetError());
    }

    win = SDL_CreateWindow(title.c_str(), x, y, w, h, flags);
    if (!win) {
      SDL_Quit();
      throw SDLError(std::string("SDL_CreateWindow: ") + SDL_GetError());
    }

    renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
//      std::cerr << "SDL_CreateRenderer error: " << SDL_GetError() << std::endl;
      SDL_DestroyWindow(win);
      SDL_Quit();
      throw SDLError(std::string("SDL_CreateRenderer: ") + SDL_GetError());
    }

    ctx = SDL_GL_CreateContext(win);
    if (ctx == nullptr) {
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(win);
      SDL_Quit();
      throw SDLError(std::string("SDL_GL_CreateContext: ") + SDL_GetError());
    }
  }

  ~SDLOpenGLContext() {
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
  }
};

float mod(float x, float y) {
  return x - std::floor(x / y) * y;
}

using color = std::tuple<int, int, int>;

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

  color worldAt(float x, float y) {
    x = std::floor(x * 10) / 10;
    y = std::floor(y * 10) / 10;
    return {
        int(mod(int(x * 255), 255)),
        int(mod(int(y * 255), 255)),
        125,
    };
  }

  color viewAt(float dx, float dy, float t) {
    t = std::sin(t);
    t = std::pow(std::abs(t), 1/1.5f) * (t < 0 ? -1 : 1);
    float rad = (dx * dx + dy * dy);
    auto fin = std::complex<float>(dx, dy) * std::pow(std::complex<float>(std::exp(1)), std::complex<float>(0, std::sqrt(rad) * 5 * t));
    float fin_dx = fin.real();
    float fin_dy = fin.imag();
    auto [r, g, b] = worldAt(pos_x + fin_dx, pos_y + fin_dy);
    float k = std::max(0.0f, 1 - (dx*dx + dy*dy));
    return {r * k, g * k, b * k};
  }
};


struct Axis {
  bool neg = false, pos = false;

  float delta() {
    return (neg * -1.0f + pos * 1.0f);
  }
};

void dow() {
  auto ctx = SDLOpenGLContext("Hello!", 100, 100, 500, 500, SDL_WINDOW_SHOWN);

  printVal(GL_RENDERER, "GL_VENDOR");
  printVal(GL_VENDOR, "GL_RENDERER");
  printVal(GL_VERSION, "GL_VERSION");

  Axis dx, dy;
  World world;



  auto prev_frame = std::chrono::steady_clock::now();
  auto start = prev_frame;
  while (true) {
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration_cast<std::chrono::duration<float, std::chrono::seconds::period>>(now - prev_frame).count();
    prev_frame = now;

    std::cout << dt << std::endl;

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

    SDL_Surface *surf = SDL_GetWindowSurface(ctx.win);


    SDL_LockSurface(surf);
    auto pixels = reinterpret_cast<uint32_t*>(surf->pixels);
    for (int y = 0; y < surf->h; ++y) {
      for (int x = 0; x < surf->w; ++x) {
        auto pix = pixels + y * surf->pitch / 4 + x;

        float k = std::min(surf->w, surf->h) / 2.0f;
        float rel_x = (x + 0.5f - surf->w / 2.0f) / k;
        float rel_y = -(y + 0.5f - surf->h / 2.0f) / k;

        auto [r, g, b] = world.viewAt(rel_x, rel_y, t);
        *pix = SDL_MapRGB(
            surf->format,
            r,
            g,
            b);
      }
    }
    SDL_UnlockSurface(surf);

//    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    SDL_UpdateWindowSurface(ctx.win);

    auto left = std::chrono::steady_clock::now() - prev_frame;
    SDL_Delay(std::max(0.0f, 1000.0f / 30 - std::chrono::duration_cast<std::chrono::milliseconds>(left).count()));
  }
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