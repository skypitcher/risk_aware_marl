import os
import numpy as np
import OpenGL.GL as gl
from PIL import Image, ImageDraw # ImageDraw might not be strictly needed but good for completeness
import requests # For downloading the texture


def _create_texture_from_image(image):
    """Helper to create an OpenGL texture from a PIL image."""
    img_data = np.array(image)
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    # gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
        image.width, image.height,
        0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data
    )
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id


class TextureManager:
    def __init__(self, assets_dir: str):
        self.assets_dir = assets_dir
        self.earth_texture_id = None
        self.earth_grayscale_texture_id = None
        # Ensure GL context is available when _load_earth_textures is called.
        # This is typically handled by ensuring glfw.make_context_current is called before TextureManager instantiation.
        self._load_earth_textures()

    def _load_earth_textures(self):
        """Load Earth map texture for background."""
        texture_path = os.path.join(self.assets_dir, "earth_map.jpg")
        
        print(f"Attempting to load Earth map from: {texture_path}") # Debug print

        if not os.path.exists(texture_path):
            print(f"Earth map not found at {texture_path}. Attempting to download...")
            # Ensure the assets directory exists before downloading
            if not os.path.exists(self.assets_dir):
                try:
                    os.makedirs(self.assets_dir)
                    print(f"Created assets directory: {self.assets_dir}")
                except OSError as e:
                    print(f"Error creating assets directory {self.assets_dir}: {e}")
                    return
            try:
                response = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Natural_Earth_III_Current_Event_Map.jpg/2560px-Natural_Earth_III_Current_Event_Map.jpg", stream=True)
                response.raise_for_status()
                with open(texture_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Download complete to {texture_path}.")
            except Exception as e:
                print(f"Error downloading Earth map to {texture_path}: {e}")
                return

        try:
            image = Image.open(texture_path)
            # Ensure image is RGB as required by OpenGL texture format
            if image.mode != "RGB":
                image = image.convert("RGB")

            print(f"Loaded image with original size: {image.width} x {image.height}")
            
            self.earth_texture_id = _create_texture_from_image(image)
            
            # Create grayscale version from the original loaded image
            grayscale_image = image.convert("L").convert("RGB")
            self.earth_grayscale_texture_id = _create_texture_from_image(grayscale_image)
            
            if self.earth_texture_id and self.earth_grayscale_texture_id:
                 print(f"Earth textures loaded successfully (Color: {self.earth_texture_id}, Grayscale: {self.earth_grayscale_texture_id}).")
            else:
                print("Failed to create one or both Earth textures.")

        except FileNotFoundError:
            print(f"Error: Earth map file not found at {texture_path} after download attempt (if any).")
            self.earth_texture_id = None
            self.earth_grayscale_texture_id = None
        except Exception as e:
            print(f"Error loading Earth texture from {texture_path}: {e}")
            self.earth_texture_id = None
            self.earth_grayscale_texture_id = None
            
    def get_texture_id(self, grayscale: bool):
        return self.earth_grayscale_texture_id if grayscale else self.earth_texture_id

    def cleanup(self):
        """Delete OpenGL textures."""
        if self.earth_texture_id is not None:
            gl.glDeleteTextures([self.earth_texture_id])
            self.earth_texture_id = None
        if self.earth_grayscale_texture_id is not None:
            gl.glDeleteTextures([self.earth_grayscale_texture_id])
            self.earth_grayscale_texture_id = None
        print("TextureManager cleaned up textures.") 