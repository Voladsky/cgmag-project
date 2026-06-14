// water_waves.cpp
// Minimal demo: animated ocean with environment reflections and moving ships
// Dependencies: GLFW, glad, Assimp, stb_image, glm

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Global constants
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;
const int WATER_GRID_SIZE = 512;
const float WATER_SCALE = 0.5f;

// Runtime state
float deltaTime = 0.0f, lastFrame = 0.0f;
glm::vec3 cameraPos = glm::vec3(0.0f, 10.0f, 80.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.3f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float yaw = -90.0f, pitch = -15.0f;
float lastX = SCR_WIDTH / 2, lastY = SCR_HEIGHT / 2;
bool firstMouse = true;

// Skybox selection and reload flag
bool useSkybox2 = true;  // select alternate skybox set
bool skyboxChanged = true; // request reload when toggled

// Ship circular motion parameters
const float SHIP_RADIUS = 30.0f;
const float SHIP_ANGULAR_SPEED = 0.15f;
const float SHIP1_INIT_ANGLE = 0.0f;
const float SHIP2_INIT_ANGLE = glm::pi<float>();
// Scale applied to loaded ship model
const float SHIP_SCALE = 0.01f;

// Forward declarations
void framebuffer_size_callback(GLFWwindow* window, int w, int h);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
unsigned int loadCubemap(std::vector<std::string> faces);
unsigned int loadTexture(const char* path);
unsigned int loadSkybox(bool useSkybox2);
class Shader;
struct WaterMesh;
struct Model;

// Small Shader helper: compiles and links GLSL sources
class Shader {
public:
    unsigned int ID;
    
    Shader(const char* vertexCode, const char* fragmentCode,
           const char* geometryCode = nullptr) {
        const char* vShaderCode = vertexCode;
        const char* fShaderCode = fragmentCode;
        const char* gShaderCode = geometryCode;

        unsigned int vertex, geometry = 0, fragment;
        
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        if (geometryCode != nullptr) {
            geometry = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometry, 1, &gShaderCode, NULL);
            glCompileShader(geometry);
            checkCompileErrors(geometry, "GEOMETRY");
        }

        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        if (geometryCode != nullptr)
            glAttachShader(ID, geometry);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");

        glDeleteShader(vertex);
        if (geometryCode != nullptr)
            glDeleteShader(geometry);
        glDeleteShader(fragment);
    }
    
    void use() { glUseProgram(ID); }
    void setMat4(const std::string &name, const glm::mat4 &mat) {
        glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    void setVec3(const std::string &name, const glm::vec3 &value) {
        glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
    }
    void setFloat(const std::string &name, float value) {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }
    void setInt(const std::string &name, int value) {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }
    
private:
    void checkCompileErrors(unsigned int shader, std::string type) {
        GLint success;
        GLchar infoLog[1024];
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
            }
        } else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "ERROR::PROGRAM_LINKING_ERROR\n" << infoLog << std::endl;
            }
        }
    }
};

// Model loader using Assimp (loads meshes and textures)
struct Model {
    struct Mesh {
        unsigned int VAO, VBO, EBO;
        unsigned int indexCount;
        unsigned int diffuseTex;
        Mesh(std::vector<glm::vec3>& positions, std::vector<glm::vec3>& normals,
            std::vector<glm::vec2>& texCoords, std::vector<unsigned int>& indices,
            unsigned int diffuse) {
            indexCount = indices.size();
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            std::vector<float> data;
            for (size_t i = 0; i < positions.size(); ++i) {
                data.push_back(positions[i].x);
                data.push_back(positions[i].y);
                data.push_back(positions[i].z);
                data.push_back(normals[i].x);
                data.push_back(normals[i].y);
                data.push_back(normals[i].z);
                if (i < texCoords.size()) {
                    data.push_back(texCoords[i].x);
                    data.push_back(texCoords[i].y);
                } else {
                    data.push_back(0.0f); data.push_back(0.0f);
                }
            }
            glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
            glBindVertexArray(0);
            diffuseTex = diffuse;
        }
        void draw(Shader& shader) {
            if (diffuseTex) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, diffuseTex);
            }
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    };

    std::vector<Mesh> meshes;
    std::map<std::string, unsigned int> texturesLoaded;

    Model(const char* path) {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
            return;
        }
        processNode(scene->mRootNode, scene);
    }

    void processNode(aiNode* node, const aiScene* scene) {
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene);
        }
    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene) {
        std::vector<glm::vec3> positions, normals;
        std::vector<glm::vec2> texCoords;
        std::vector<unsigned int> indices;

        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            positions.push_back(glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
            normals.push_back(glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
            if (mesh->mTextureCoords[0]) {
                texCoords.push_back(glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y));
            } else {
                texCoords.push_back(glm::vec2(0.0f));
            }
        }
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        unsigned int diffuseMap = 0;
        if (mesh->mMaterialIndex >= 0) {
            aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];
            aiString str;
            if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &str) == AI_SUCCESS) {
                std::string fullPath = std::string("models/") + str.C_Str();
                if (texturesLoaded.find(fullPath) == texturesLoaded.end()) {
                    unsigned int tex = loadTexture(fullPath.c_str());
                    texturesLoaded[fullPath] = tex;
                }
                diffuseMap = texturesLoaded[fullPath];
            }
        }
        return Mesh(positions, normals, texCoords, indices, diffuseMap);
    }

    void draw(Shader& shader) {
        for (auto& mesh : meshes)
            mesh.draw(shader);
    }
};

// Water grid mesh: 2D grid of x,z positions used by geometry shader
struct WaterMesh {
    unsigned int VAO, VBO, EBO;
    unsigned int indexCount;
    WaterMesh(int gridSize, float scale) {
        std::vector<float> vertices;
        std::vector<unsigned int> indices;

        for (int i = 0; i <= gridSize; ++i) {
            for (int j = 0; j <= gridSize; ++j) {
                float x = (j - gridSize / 2.0f) * scale;
                float z = (i - gridSize / 2.0f) * scale;
                vertices.push_back(x);
                vertices.push_back(z);
            }
        }

        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                int a = i * (gridSize + 1) + j;
                int b = i * (gridSize + 1) + j + 1;
                int c = (i + 1) * (gridSize + 1) + j;
                int d = (i + 1) * (gridSize + 1) + j + 1;
                indices.push_back(a); indices.push_back(c); indices.push_back(b);
                indices.push_back(b); indices.push_back(c); indices.push_back(d);
            }
        }
        indexCount = indices.size();

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glBindVertexArray(0);
    }
    void draw() {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

// Texture loading helpers (stb_image)
unsigned int loadTexture(const char* path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1) format = GL_RED;
        else if (nrComponents == 3) format = GL_RGB;
        else if (nrComponents == 4) format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    } else {
        std::cerr << "Texture failed to load: " << path << std::endl;
        stbi_image_free(data);
    }
    return textureID;
}

unsigned int loadCubemap(std::vector<std::string> faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    for (unsigned int i = 0; i < faces.size(); i++) {
        unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            unsigned char* flippedData = new unsigned char[width * height * nrChannels];
            int rowSize = width * nrChannels;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    memcpy(flippedData + y * rowSize + (width - 1 - x) * nrChannels,
                           data + y * rowSize + x * nrChannels,
                           nrChannels);
                }
            }
            
            GLenum format;
            if (nrChannels == 1) format = GL_RED;
            else if (nrChannels == 3) format = GL_RGB;
            else if (nrChannels == 4) format = GL_RGBA;
            
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, width, height, 0, 
                        format, GL_UNSIGNED_BYTE, flippedData);
            
            delete[] flippedData;
            stbi_image_free(data);
        } else {
            std::cerr << "Cubemap failed to load: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }
    
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    return textureID;
}

// Load a cubemap from one of two skybox folders
unsigned int loadSkybox(bool useSkybox2) {
    std::vector<std::string> skyboxFaces;
    std::string folder = useSkybox2 ? "skybox2" : "skybox";
    
    std::cout << "Loading skybox from folder: " << folder << std::endl;
    
    skyboxFaces.push_back("textures/" + folder + "/right.png");
    skyboxFaces.push_back("textures/" + folder + "/left.png");
    skyboxFaces.push_back("textures/" + folder + "/top.png");
    skyboxFaces.push_back("textures/" + folder + "/bottom.png");
    skyboxFaces.push_back("textures/" + folder + "/front.png");
    skyboxFaces.push_back("textures/" + folder + "/back.png");
    
    return loadCubemap(skyboxFaces);
}

// GLSL shader sources
const char* waterVertexShader = R"(
// Vertex shader for water grid
// - Input: 2D grid coordinate (x,z) in object space
// - Output: pass-through texture coordinates for normal mapping
#version 330 core
layout (location = 0) in vec2 aPos; // x (world-x), z (world-z)
out vec2 vTexCoords; // to be consumed by geometry shader
void main() {
    // Place vertex in clip-space with y=0; geometry shader will do displacement
    gl_Position = vec4(aPos.x, 0.0, aPos.y, 1.0);
    // Small-scale UVs used for sampling normal map and foam
    vTexCoords = aPos * 0.05 + vec2(0.5);
}
)";

const char* waterGeometryShader = R"(
// Geometry shader: displaces flat grid into animated ocean surface
// - Receives triangle vertices in object (x,z) plane
// - Computes Gerstner-like multi-wave displacement per-vertex
// - Emits world-space position, normal, UV, height and foam factor
#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform float uTime; // global time for animation
uniform mat4 uProjection; // camera projection
uniform mat4 uView; // camera view
uniform float uCalmness; // global scale for wave amplitudes/steepness

in vec2 vTexCoords[]; // UVs from vertex shader

out vec3 FragPos; // world-space position for fragment lighting/reflection
out vec3 Normal;  // displaced normal
out vec2 TexCoords; // UVs for normal/foam textures
out float Height; // world-space height for foam calculation
out float FoamFactor; // foam intensity [0..1]

const float PI = 3.14159265359;

// Simple wave descriptor used to sum multiple components
struct Wave {
    vec2 dir;
    float amplitude;
    float wavelength;
    float steepness;
};

// Composition of multiple directional waves produces rich surface
const Wave waves[8] = Wave[](
    Wave(vec2(1.0, 0.0),      0.4, 10.0, 0.7),
    Wave(vec2(0.9, 0.1),      0.3,  8.0, 0.65),
    Wave(vec2(0.95, -0.05),   0.25, 7.0, 0.6),
    Wave(vec2(0.85, 0.15),    0.2, 11.0, 0.5),
    Wave(vec2(-0.8, 0.3),     0.2,  9.0, 0.55),
    Wave(vec2(0.7, 0.2),      0.15, 6.5, 0.5),
    Wave(vec2(-0.9, 0.05),    0.15, 10.5, 0.4),
    Wave(vec2(0.6, -0.3),     0.1,  6.0, 0.4)
);

// Deep-water dispersion approximation for angular frequency
float waveOmega(float L) {
    return sqrt(9.81 * 2.0 * PI / L);
}

// Displace a 2D point (x,z) into a 3D position using summed waves.
// Returns vec3(worldX, worldY, worldZ).
vec3 displace(vec2 p) {
    float x = p.x;
    float z = p.y;
    float y = 0.0;
    float dx = 0.0;
    float dz = 0.0;

    // Sum multiple directional components (Gerstner-like offsets)
    for (int i = 0; i < 8; i++) {
        vec2 D = normalize(waves[i].dir);
        float A = waves[i].amplitude * uCalmness;
        float L = waves[i].wavelength;
        float Q = waves[i].steepness * uCalmness;
        float w = 2.0 * PI / L;
        float omega = waveOmega(L);
        float phase = w * dot(D, vec2(x, z)) - omega * uTime;
        float sinP = sin(phase);
        float cosP = cos(phase);

        // vertical displacement
        y += A * cosP;
        // horizontal offsets used for normal calculation (reduce choppiness with Q)
        dx -= Q * A * D.x * sinP;
        dz -= Q * A * D.y * sinP;
    }
    return vec3(x + dx, y, z + dz);
}

void main() {
    // Small finite-difference step for normal estimation
    float eps = 0.1;

    for (int i = 0; i < 3; i++) {
        vec3 base = gl_in[i].gl_Position.xyz; // input x,z plane
        vec2 xz = vec2(base.x, base.z);

        // compute displaced vertex and nearby samples for normal
        vec3 worldPos = displace(xz);
        vec3 posR = displace(vec2(xz.x + eps, xz.y));
        vec3 posL = displace(vec2(xz.x - eps, xz.y));
        vec3 posU = displace(vec2(xz.x, xz.y + eps));
        vec3 posD = displace(vec2(xz.x, xz.y - eps));

        // tangent vectors and normal via cross product
        vec3 tangentX = posR - posL;
        vec3 tangentZ = posU - posD;
        vec3 norm = normalize(cross(tangentX, tangentZ));
        if (norm.y < 0.0) norm = -norm; // keep normal pointing up

        // Transform to clip space and emit attributes
        gl_Position = uProjection * uView * vec4(worldPos, 1.0);
        FragPos = worldPos;
        Normal = norm;
        TexCoords = vTexCoords[i];
        Height = worldPos.y;

        // Foam factor: smoothstep gives soft edges for foam regions
        FoamFactor = smoothstep(0.3, 0.8, worldPos.y);

        EmitVertex();
    }
    EndPrimitive();
}
)";

const char* waterFragmentShader = R"(
// Fragment shader for water surface
// - Uses normal map perturbation + geometry normal for small-scale detail
// - Computes view-dependent reflection via environment cubemap (skybox)
// - Applies Fresnel blend between base water color and reflection
// - Adds foam using height-based factor provided by geometry shader
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in float Height;
in float FoamFactor;

out vec4 FragColor;

uniform samplerCube skybox;    // environment for reflections
uniform sampler2D normalMap;  // tileable normal map for surface detail
uniform vec3 viewPos;         // camera position in world space
uniform float uTime;          // animated offset for normal sampling

void main() {
    // base normal from geometry, then perturb by normal map for small-scale ripples
    vec3 N = normalize(Normal);
    vec3 normalSample = texture(normalMap, TexCoords * 10.0 + uTime * 0.05).rgb;
    vec3 perturbedNormal = normalize(N + (normalSample * 2.0 - 1.0) * 0.2);

    // view and reflection directions
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-viewDir, perturbedNormal);
    vec3 reflectColor = texture(skybox, reflectDir).rgb;

    // Fresnel (Schlick-esque) to blend between water base color and reflection
    float fresnel = pow(1.0 - max(dot(viewDir, perturbedNormal), 0.0), 3.0);

    vec3 waterColor = vec3(0.0, 0.3, 0.5); // subsurface tint
    vec3 color = mix(waterColor, reflectColor, fresnel);

    // foam: blend towards white where geometry signaled breaking crests
    vec3 foamColor = vec3(1.0);
    color = mix(color, foamColor, FoamFactor * 0.8);

    // final alpha slightly translucent for subtle blending with background
    FragColor = vec4(color, 0.9);
}
)";

const char* skyboxVertexShader = R"(
// Skybox vertex shader: render a unit cube in clip-space
// Trick: replicate depth at far plane by writing pos.xyww
#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 TexCoords;
uniform mat4 projection;
uniform mat4 view;
void main() {
    TexCoords = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww; // force depth to far plane
}
)";

const char* skyboxFragmentShader = R"(
// Sample the cubemap with the interpolated direction
#version 330 core
in vec3 TexCoords;
out vec4 FragColor;
uniform samplerCube skybox;
void main() {
    FragColor = texture(skybox, TexCoords);
}
)";

const char* shipVertexShader = R"(
// Simple ship vertex shader with model-space to world-space transform
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal; // correct normal transform
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* shipFragmentShader = R"(
// Simple textured diffuse shader for ships
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture_diffuse1; // base texture
uniform vec3 lightPos; // world-space light

void main() {
    vec3 color = texture(texture_diffuse1, TexCoord).rgb;
    vec3 ambient = 0.2 * color;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;
    FragColor = vec4(ambient + diffuse, 1.0);
}
)";

// -------------------- Функция отрисовки корабля --------------------
void drawShip(Shader& shader, const glm::mat4& view, const glm::mat4& projection,
              const glm::vec3& position, const glm::vec3& forward, Model& ship) {
    float yawAngle = glm::pi<float>() + atan2(forward.x, forward.z);
    glm::quat yawQuat = glm::angleAxis(yawAngle, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat tilt = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = model * glm::mat4_cast(yawQuat) * glm::mat4_cast(tilt);
    model = glm::scale(model, glm::vec3(SHIP_SCALE));

    shader.setMat4("model", model);
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);
    ship.draw(shader);
}

// -------------------- Основной код --------------------
int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Water Waves - Press 'B' to switch skybox", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Загрузка ресурсов
    unsigned int normalMap = loadTexture("textures/water_normal.jpg");
    unsigned int skyboxTexture = loadSkybox(useSkybox2);

    Shader waterShader(waterVertexShader, waterFragmentShader, waterGeometryShader);
    Shader skyboxShader(skyboxVertexShader, skyboxFragmentShader);
    Shader shipShader(shipVertexShader, shipFragmentShader);

    WaterMesh water(WATER_GRID_SIZE, WATER_SCALE);
    Model ship("models/gabarre.obj");

    float skyboxVertices[] = {
        -1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f
    };
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    glm::vec3 lightPos(5.0f, 10.0f, 5.0f);

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);

        // Reload skybox if needed
        if (skyboxChanged) {
            unsigned int newSkyboxTexture = loadSkybox(useSkybox2);
            if (newSkyboxTexture) {
                glDeleteTextures(1, &skyboxTexture);
                skyboxTexture = newSkyboxTexture;
                std::cout << "Skybox switched to: " << (useSkybox2 ? "skybox2" : "skybox") << std::endl;
            }
            skyboxChanged = false;
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        // 1. Вода
        waterShader.use();
        waterShader.setMat4("uProjection", projection);
        waterShader.setMat4("uView", view);
        waterShader.setFloat("uTime", currentFrame);
        // Apply calmer waves when using skybox2
        float calmness = useSkybox2 ? 0.4f : 1.0f;
        waterShader.setFloat("uCalmness", calmness);
        waterShader.setVec3("viewPos", cameraPos);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, normalMap);
        waterShader.setInt("normalMap", 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
        waterShader.setInt("skybox", 1);
        water.draw();

        // 2. Скайбокс
        glDepthFunc(GL_LEQUAL);
        skyboxShader.use();
        skyboxShader.setMat4("projection", projection);
        skyboxShader.setMat4("view", glm::mat4(glm::mat3(view)));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS);

        // 3. Корабли
        shipShader.use();
        shipShader.setVec3("lightPos", lightPos);

        float angle1 = SHIP1_INIT_ANGLE + SHIP_ANGULAR_SPEED * currentFrame;
        float x1 = SHIP_RADIUS * cos(angle1);
        float z1 = SHIP_RADIUS * sin(angle1);
        glm::vec3 pos1 = glm::vec3(x1, 0.0f, z1);
        glm::vec3 tangent1 = glm::normalize(glm::vec3(-sin(angle1), 0.0f, cos(angle1)));
        drawShip(shipShader, view, projection, pos1, tangent1, ship);

        float angle2 = SHIP2_INIT_ANGLE + SHIP_ANGULAR_SPEED * currentFrame;
        float x2 = SHIP_RADIUS * cos(angle2);
        float z2 = SHIP_RADIUS * sin(angle2);
        glm::vec3 pos2 = glm::vec3(x2, 0.0f, z2);
        glm::vec3 tangent2 = glm::normalize(glm::vec3(-sin(angle2), 0.0f, cos(angle2)));
        drawShip(shipShader, view, projection, pos2, tangent2, ship);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

// -------------------- Обработчики ввода --------------------
void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

void mouse_callback(GLFWwindow*, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    float sensitivity = 0.1f;
    xoffset *= sensitivity; yoffset *= sensitivity;
    yaw += xoffset; pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void scroll_callback(GLFWwindow*, double, double) {
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_B:
                // Switch between skybox and skybox2
                useSkybox2 = !useSkybox2;
                skyboxChanged = true;
                std::cout << "Switching skybox to: " << (useSkybox2 ? "skybox2" : "skybox") << std::endl;
                break;
            
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, true);
                break;
        }
    }
}

void processInput(GLFWwindow* window) {
    float speed = 5.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
}