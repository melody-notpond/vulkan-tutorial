CXX=clang++
CFLAGS=-g -lvulkan -lglfw -std=c++20 -Wall -Wpedantic -Werror
SFLAGS=-target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vertMain -entry fragMain
SHADERSC=_shaders.cpp
SPV=slang.spv
TARGET=vulk

$(TARGET): src/*.cpp $(SHADERSC)
	$(CXX) $(CFLAGS) $? -o $(TARGET)

$(SHADERSC): $(SPV)
	xxd -i -n shaders $? > $(SHADERSC)

$(SPV): shaders/shader.slang
	slangc $? $(SFLAGS) -o $(SPV)

run: $(TARGET)
	./$(TARGET)

clean:
	@rm *.spv $(TARGET) $(SHADERSC) 2>/dev/null || true
