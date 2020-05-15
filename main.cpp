#include <iostream>
#include <SDL.h>
#include <GL/glew.h>
#include <imgui.h>

void printVal(GLenum tp, std::string name) {
  const GLubyte* sv = glGetString(tp);
  if (sv == nullptr) {
    std::cerr << "can't get " << name << ": " << glewGetErrorString(glGetError()) << std::endl;
  } else {
    std::cerr << name << ": " << reinterpret_cast<const char*>(sv) << std::endl;
  }
}

int main(int, char**){
  if (SDL_Init(SDL_INIT_VIDEO) != 0){
    std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;
    return 1;
  }

  SDL_Window *win = SDL_CreateWindow("Hello!", 100, 100, 800, 600, SDL_WINDOW_SHOWN);
  if (!win) {
    std::cerr << "SDL_CreateWindow error: " << SDL_GetError() << std::endl;
    SDL_Quit();
  }

  SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (!renderer) {
    std::cerr << "SDL_CreateRenderer error: " << SDL_GetError() << std::endl;
    SDL_DestroyWindow(win);
    SDL_Quit();
  }

  SDL_GLContext ctx = SDL_GL_CreateContext(win);


  printVal(GL_RENDERER, "GL_VENDOR");
  printVal(GL_VENDOR, "GL_RENDERER");
  printVal(GL_VERSION, "GL_VERSION");

//  char c;
//  std::cin >> c;

//  ImGui::BeginMenu("", false);

  SDL_GL_DeleteContext(ctx); ctx = nullptr;

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}