import pygame
import numpy as np


class SokobanMap:
    _image_names = [
        'floor', 'wall', 'crate', 'goal', 'goal_crate',
        'up', 'up1', 'up2',
        'down', 'down1', 'down2',
        'left', 'left1', 'left2',
        'right', 'right1', 'right2'
    ]

    _tile_name = ['floor', 'wall', 'crate', 'goal', 'goal_crate', 'player']

    _tile_code = {
        'floor': 0,
        'wall': 1,
        'crate': 2,
        'goal': 3,
        'goal_crate': 4,
        'player': 5
    }

    _tile_to_image = {
        'floor': 'floor',
        'wall': 'wall',
        'crate': 'crate',
        'goal': 'goal',
        'goal_crate': 'goal_crate',
        'player': 'down'
    }

    # Initialized the map with the given map_template - ID (0 - 999) or numpy array.
    def __init__(self, map_template=None, scale=None, dir=''):
        self._images = None
        self._tile_size = None
        self._map = None
        self._map_size = None
        self._player_position = None
        self._player_direction = None
        self._initial_state = None
        self._animation = None

        self._load_images(scale, dir)

        if map_template is None:
            map_template = np.random.randint(1000)

        if isinstance(map_template, int):
            if map_template >= 0 and map_template <= 999:
                self._load_map(map_template, dir)
                self._process_map()

        elif isinstance(map_template, np.ndarray):
            if map_template.ndim == 2:
                (map_height, map_width) = map_template.shape

                if map_width > 1 and map_height > 1:
                    self._map = np.array(map_template, dtype=np.uint8)
                    self._process_map()

        # If no map is loaded, create a trivial one.
        if self._map is None:
            self._map = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.uint8)
            self._process_map()

        self._initial_state = {
            'map': np.copy(self._map),
            'player': {
                'position': self._player_position,
                'direction': self._player_direction
            }
        }

    # Load the tiles from PNG files.
    def _load_images(self, scale=None, dir=''):
        self._tile_size = None

        # Iterate through all the tiles.
        self._images = {}
        for image_name in self._image_names:
            # Load the PNG file.
            img = pygame.image.load(f'{dir}/img/{image_name}.png')
            
            # The size of the tile has to match other tiles.
            if self._tile_size is None:
                (img_width, img_height) = self._tile_size = img.get_size()
                
                # If scaling, set the new tile size.
                if scale is not None:
                    (img_width, img_height) = (round(img_width * scale), round(img_height * scale))
            else:
                assert self._tile_size == img.get_size()

            # Store the tile in the original size or scale it.
            if scale is None:
                self._images[image_name] = img
            else:
                self._images[image_name] = pygame.transform.scale(img,  (img_width, img_height))
        
        # Set the tile size globally.
        self._tile_size = (img_width, img_height)

    # Load the map with the given ID from the levels.txt file.
    def _load_map(self, id, dir):
        # Read the levels file.
        f = open(f'{dir}/levels.txt', 'r')
        lines = f.readlines()
        f.close()

        # Find and load the map from the file.
        i = 0
        found = False
        loaded = False
        while not (found and loaded) and i < len(lines):
            # Load a line from the file.
            line = lines[i].strip()

            # If we are still searching for the map.
            if not found:
                if len(line) > 0 and line[0] == ';':
                    if int(line.split(' ')[1]) == id:
                        found = True
                        self._map = []
            
            # If we are loading the map:
            else:
                # If all the map lines have been read.
                if len(line) == 0:
                    loaded = True
                    continue

                # If another row is being loaded.
                row = []
                for c in line:
                    row.append(int(c))
                self._map.append(row)

            i += 1

        # Convert the map to numpy array.
        self._map = np.array(self._map, dtype=np.uint8)

    def _process_map(self):
        if self._map is None:
            return

        (map_height, map_width) = self._map.shape
        self._map_size = (map_width, map_height)

        # Find and separate the player.
        for y in range(map_height):
            for x in range(map_width):
                if self._map[y][x] == self._tile_code['player']:
                    self._map[y][x] = self._tile_code['floor']
                    self._player_position = (x, y)
                    self._player_direction = 'down'

        # Remove excess walls.
        remove = []
        for y in range(map_height):
            for x in range(map_width):
                if self._map[y][x] == self._tile_code['wall']:
                    all_walls = True
                    for dy in range(3):
                        for dx in range(3):
                            x1 = x + dx - 1
                            y1 = y + dy - 1
                            if x1 >= 0 and x1 < map_width and y1 >= 0 and y1 < map_height:
                                if self._map[y1][x1] != self._tile_code['wall']:
                                    all_walls = False
                    if all_walls:
                        remove.append((x, y))
        
        for (x, y) in remove:
            self._map[y][x] = self._tile_code['floor']
        
        if self.player_position is None:
            self._map = None

    # Return the window size in pixels.
    def window_size(self):
        if self._tile_size is None or self._map_size is None:
            return (640, 640)
        
        (tile_width, tile_height) = self._tile_size
        (map_width, map_height) = self._map_size

        return (tile_width * map_width, tile_height * map_height)

    # Reset the map.
    def reset(self):
        if self._initial_state is None:
            return
        
        self._map = np.copy(self._initial_state['map'])
        self._player_position = self._initial_state['player']['position']
        self._player_direction = self._initial_state['player']['direction']

    # Return the current player position.
    def player_position(self):
        return self._player_position

    # Return the map.
    def get_array(self):
        map = np.copy(self._map)
        (x, y) = self._player_position
        map[y][x] = self._tile_code['player']
        return map

    # Return the maps dimensions.
    def get_map_size(self):
        return self._map_size

    # Check if game is finished.
    def game_finished(self):
        # Find a crate that is not on a goal position.
        for row in self._map:
            for cell in row:
                if cell == self._tile_code['crate']:
                    return False
        return True

    # Start animating player motion.
    def move_player(self, dx, dy, animate=False, speed=4.0/60):
        # If no player, do nothing.
        if self._player_position is None:
            return

        # If animation is running, stop the animation first.
        if self._animation is not None:
            self.stop_animation()

        (map_width, map_height) = self._map_size
        (tile_width, tile_height) = self._tile_size
        (x, y) = self._player_position

        # Turn the player.
        if dx == 1:
            self._player_direction = 'right'
        elif dx == -1:
            self._player_direction = 'left'
        elif dy == 1:
            self._player_direction = 'down'
        elif dy == -1:
            self._player_direction = 'up'
        else:
            return

        # Check if motion possible.
        x1 = x + dx
        y1 = y + dy

        # Edge of the map.
        if x1 < 0 or x1 >= map_width or y1 < 0 or y1 >= map_height:
            return
        
        # Moving the player to an empty tile.
        if self._map[y1][x1] == self._tile_code['floor'] or self._map[y1][x1] == self._tile_code['goal']:
            if animate:
                # Set up the animation.
                animation_player = {
                    'source_tile': (x, y),
                    'direction': (dx, dy),
                    'pixel_position': (x * tile_width, y * tile_height),
                    'frame': 0
                }

                self._animation = {
                    'speed': speed,
                    'progress': 0,
                    'player': animation_player,
                    'crate': None
                }
            else:
                # Set the new player position.
                self._player_position = (x1, y1)

            return

        # Pushing a crate
        if self._map[y1][x1] == self._tile_code['crate'] or self._map[y1][x1] == self._tile_code['goal_crate']:
            # Check if crate can be pushed.
            x2 = x1 + dx
            y2 = y1 + dy

            # Edge of the map.
            if x2 < 0 or x2 >= map_width or y2 < 0 or y2 >= map_height:
                return

            # Moving the crate to an empty tile.
            if self._map[y2][x2] == self._tile_code['floor'] or self._map[y2][x2] == self._tile_code['goal']:
                # Remove the crate from the map.
                if self._map[y1][x1] == self._tile_code['crate']:
                    self._map[y1][x1] = self._tile_code['floor']
                elif self._map[y1][x1] == self._tile_code['goal_crate']:
                    self._map[y1][x1] = self._tile_code['goal']
                else:
                    return

                if animate:
                    # Set up the animation.
                    animation_player = {
                        'source_tile': (x, y),
                        'direction': (dx, dy),
                        'pixel_position': (x * tile_width, y * tile_height),
                        'frame': 0
                    }

                    animation_crate = {
                        'source_tile': (x1, y1),
                        'direction': (dx, dy),
                        'pixel_position': (x1 * tile_width, y1 * tile_height),
                    }

                    self._animation = {
                        'speed': speed,
                        'progress': 0,
                        'player': animation_player,
                        'crate': animation_crate
                    }
                else:
                    # Set the new player position.
                    self._player_position = (x1, y1)

                    # Place the box on the map.
                    if self._map[y2][x2] == self._tile_code['floor']:
                        self._map[y2][x2] = self._tile_code['crate']
                    elif self._map[y2][x2] == self._tile_code['goal']:
                        self._map[y2][x2] = self._tile_code['goal_crate']

    # Make one animation step.
    def animate_step(self):
        if self._animation is None:
            return

        # Get tile size.
        (tile_width, tile_height) = self._tile_size

        # increase the animation progress.
        self._animation['progress'] += self._animation['speed']
        progress = self._animation['progress']
        
        # Move the player.
        (x0, y0) = self._animation['player']['source_tile']
        (dx, dy) = self._animation['player']['direction']
        self._animation['player']['pixel_position'] = (
            round((x0 + dx * progress) * tile_width),
            round((y0 + dy * progress) * tile_height)
        )
        
        # Set the player's frame.
        if progress >= 0.0 and progress < 0.25:
            self._animation['player']['frame'] = 1
        elif progress >= 0.25 and progress < 0.5:
            self._animation['player']['frame'] = 2
        elif progress >= 0.5 and progress < 0.75:
            self._animation['player']['frame'] = 1
        elif progress >= 0.75 and progress < 1.0:
            self._animation['player']['frame'] = 2
        else:
            self._animation['player']['frame'] = 0
        
        # Move the crate.
        if self._animation['crate'] is not None:
            (x0, y0) = self._animation['crate']['source_tile']
            (dx, dy) = self._animation['crate']['direction']
            self._animation['crate']['pixel_position'] = (
                round((x0 + dx * progress) * tile_width),
                round((y0 + dy * progress) * tile_height)
            )

        # Check if animation ran out.
        if progress >= 1.0:
            # Stop the animation.
            self.stop_animation()

    # Stop the animation.
    def stop_animation(self):
        if self._animation is None:
            return

        # Set the new player position.
        (x0, y0) = self._animation['player']['source_tile']
        (dx, dy) = self._animation['player']['direction']
        self._player_position = (x0 + dx, y0 + dy)

        # Put the crate on the map.
        if self._animation['crate'] is not None:
            (x0, y0) = self._animation['crate']['source_tile']
            (dx, dy) = self._animation['crate']['direction']
            (x, y) = (x0 + dx, y0 + dy)

            if self._map[y][x] == self._tile_code['floor']:
                self._map[y][x] = self._tile_code['crate']
            elif self._map[y][x] == self._tile_code['goal']:
                self._map[y][x] = self._tile_code['goal_crate']

        # Stop the animation.
        self._animation = None

    # Is animation running?
    def animation_running(self):
        return self._animation is not None

    # Paint the map on a pygame surface.
    def paint(self, surface):
        (tile_width, tile_height) = self._tile_size
        (map_width, map_height) = self._map_size

        # Render the tiles.
        for y in range(map_height):
            for x in range(map_width):
                tile_code = self._map[y][x]
                tile_name = self._tile_name[tile_code]

                rect = pygame.Rect((x * tile_width, y * tile_height), self._tile_size)
                if tile_name != 'floor':
                    surface.blit(self._images['floor'], rect)
                
                surface.blit(self._images[self._tile_to_image[tile_name]], rect)

        # If no player, just render the tiles
        if self._player_position is None:
            return

        # Render the animated objects.
        if self._animation is not None:
            # Render the player.
            (x, y) = self._animation['player']['pixel_position']
            frame = self._animation['player']['frame']
            rect = pygame.Rect((x, y), self._tile_size)
            
            if frame == 0:
                surface.blit(self._images[f'{self._player_direction}'], rect)
            else:
                surface.blit(self._images[f'{self._player_direction}{frame}'], rect)

            # Render the crate.
            if self._animation['crate'] is not None:
                (x, y) = self._animation['crate']['pixel_position']
                rect = pygame.Rect((x, y), self._tile_size)
                surface.blit(self._images['crate'], rect)
        
        # If no animation, just render the player.
        else:
            (x, y) = self._player_position
            rect = pygame.Rect((x * tile_width, y * tile_height), self._tile_size)
            surface.blit(self._images[self._player_direction], rect)