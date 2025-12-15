CXX=clang++
CFLAGS=-g -lvulkan -lglfw -std=c++20 -Wall -Wpedantic -Werror
SFLAGS=-target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vert_main -entry frag_main
SPV=slang.spv
TARGET=vulk

$(TARGET): _shaders.cpp src/*.cpp
	$(CXX) $(CFLAGS) _shaders.cpp src/*.cpp -o $(TARGET)

_shaders.cpp: $(SPV)
	xxd -i -n shaders $? > _shaders.cpp

$(SPV): shaders/shader.slang
	slangc $? $(SFLAGS) -o $(SPV)

run: $(TARGET)
	./$(TARGET)

clean:
	@rm *.spv $(TARGET) _shaders.cpp 2>/dev/null || true
