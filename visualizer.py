import math
import sys
from typing import Union

import numpy as np
import pygame
from pygame import gfxdraw

from constants import *
from graph import Graph


class Visualizer:
    def __init__(self, drawing_lock):
        self.exit_flag = False

        self.graph = None
        self.drawing_lock = drawing_lock
        self.is_search_graph = False
        self.draw_necessary = False

        # setup pygame
        pygame.init()
        self.clock = pygame.time.Clock()

        # setup screen
        self.screen_size = SCREEN_SIZE
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        pygame.display.set_caption("Shortest Path Multicut")
        self.screen.fill(WHITE)
        self.surface = pygame.Surface(self.screen_size)
        self.surface.fill(WHITE)

        # setup text
        self.font = pygame.font.SysFont('Ariel', 32)
        waiting_text = self.font.render("Waiting for graph", True, BLACK)
        waiting_text_rec = waiting_text.get_rect()
        waiting_text_rec.center = tuple(np.round((np.array(self.screen_size) / 2)).astype(int))

        # draw init screen
        self.screen.blit(waiting_text, waiting_text_rec)
        pygame.display.update()

        self.scale = 1
        self.offset = (0, 0)
        self.set_scale()

        self.multicut = []

    def run(self):
        while True:
            if self.exit_flag:
                self.quit()

            events = pygame.event.get()
            pygame.event.pump()
            for event in events:
                if event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_size = event.dict['size']
                    self.surface = pygame.Surface(self.screen_size)
                    self.surface.fill(COLOR_KEY)
                    self.set_scale()
                    self.draw_necessary = True

            if self.graph is not None:

                if self.draw_necessary:
                    self.surface.fill(COLOR_KEY)

                    with self.drawing_lock:
                        self.draw_graph()

                    self.screen.fill(WHITE)
                    self.screen.blit(self.surface, (0, 0))
                    self.draw_necessary = False

                pygame.display.update()

            self.clock.tick(FRAMES_PER_SECOND)

    def quit(self):
        pygame.quit()
        sys.exit()

    def set_graph(self, graph: Graph):
        with self.drawing_lock:
            self.graph = graph
            self.is_search_graph = self.graph.has_node(-1)
            self.set_scale()
            self.draw_necessary = True

    def set_multicut(self, multicut):
        self.multicut = multicut
        self.draw_necessary = True

    def set_scale(self):
        if self.graph is None:
            return

        increase = 0
        if self.is_search_graph:
            increase = 2

        max_x = max(map(lambda x: x[1][0], self.graph.nodes(data="pos"))) + increase
        max_y = max(map(lambda x: x[1][1], self.graph.nodes(data="pos"))) + increase
        width = self.screen_size[0]
        height = self.screen_size[1]

        if max_x / max_y > width / height:
            # x and width is limiting
            margin = width * 0.05
            graph_width = width - margin * 2
            self.scale = graph_width / max_x
            self.offset = np.array((margin, (height - max_y * self.scale) / 2))
        else:
            # y and height is limiting
            margin = height * 0.05
            graph_height = height - margin * 2
            self.scale = graph_height / max_y
            self.offset = np.array(((width - max_x * self.scale) / 2, margin))

    def get_outside_pos(self, pos, max_x, max_y):
        pos2 = np.array(pos)
        if tuple(pos) == (0, 0):
            return [(-1, 0), (0, -1)]
        elif tuple(pos) == (max_x, 0):
            return [(max_x + 1, 0), (max_x, -1)]
        elif tuple(pos) == (0, max_y):
            return [(-1, max_y), (0, max_y + 1)]
        elif tuple(pos) == (max_x, max_y):
            return [(max_x + 1, max_y), (max_x, max_y + 1)]
        elif pos[0] == 0:
            pos2[0] -= 1
            return [pos2]
        elif pos[0] == max_x:
            pos2[0] += 1
            return [pos2]
        elif pos[1] == 0:
            pos2[1] -= 1
            return [pos2]
        elif pos[1] == max_x:
            pos2[1] += 1
            return [pos2]

        return []

    def draw_graph(self):
        self.surface.fill(COLOR_KEY)
        node_radius = self.scale_value(GRAPH_NODE_RADIUS)
        edge_width = self.scale_value(GRAPH_EDGE_WIDTH)
        num_edges = len(self.graph.edges)

        pos_offset = np.array((0, 0))
        if self.is_search_graph:
            pos_offset = np.array((1, 1))

        max_x, max_y = tuple(self.graph.get_max_pos())
        outside_positions = []

        # draw edges
        for edge in self.graph.edges(keys=True, data=True):
            pos1 = self.graph.nodes[edge[0]]['pos']
            pos2 = self.graph.nodes[edge[1]]['pos']
            if edge[1] == -1:
                positions = self.get_outside_pos(pos1, max_x, max_y)
                pos2 = positions[0]
                if tuple(pos2) in outside_positions:
                    pos2 = positions[1]
                outside_positions.append(tuple(pos2))

            cost = edge[3]["cost"]
            cut = edge[3]["id"] in self.multicut or (edge[1], edge[0], edge[2]) in self.multicut
            color = PURPLE if cut else GREEN if cost == 1 else RED
            if num_edges < 1000:
                self.draw_thick_aaline(self.scale_value(pos1 + pos_offset[0]),
                                       self.scale_value(pos2 + pos_offset[1]),
                                       color,
                                       edge_width)
            else:
                pygame.draw.line(self.surface, color,
                                 self.scale_value(pos1 + pos_offset[0]),
                                 self.scale_value(pos2 + pos_offset[1]))

        if node_radius < 5:
            return
        # draw nodes
        for node, data in self.graph.nodes(data=True):
            if node == -1:
                continue
            pos = tuple(self.scale_value(data["pos"] + pos_offset))
            color = data.get("color", DARK_BLUE)
            gfxdraw.aacircle(self.surface, *pos, node_radius, color)
            gfxdraw.filled_circle(self.surface, *pos, node_radius, color)

            cost = data.get("cost", None)
            if cost is not None:
                text_surface = self.font.render(str(cost), True, BLACK)
                text_rec = text_surface.get_rect()
                text_rec.center = pos
                self.surface.blit(text_surface, text_rec)

        # draw outside nodes
        if self.is_search_graph:
            data = self.graph.nodes[-1]
            cost = data.get("cost", 0)
            color = data.get("color", DARK_BLUE)

            if cost is not None:
                text_surface = self.font.render(str(cost), True, BLACK)
                text_rec = text_surface.get_rect()
            for pos in outside_positions:
                pos = tuple(self.scale_value(np.array(pos) + pos_offset))
                gfxdraw.aacircle(self.surface, *pos, node_radius, color)
                gfxdraw.filled_circle(self.surface, *pos, node_radius, color)
                if cost is not None:
                    text_rec.center = pos
                    self.surface.blit(text_surface, text_rec)

    def scale_value(self, value: Union[np.ndarray, float, int]):
        if value.__class__ == np.ndarray:
            return np.round(value * self.scale + self.offset).astype(int)
        else:
            result = round(value * self.scale)
            return result if result > 0 else 1

    # draw an anti-aliased line with thickness more than 1px
    # reference: https://stackoverflow.com/a/30599392
    def draw_thick_aaline(self, pos1: np.ndarray, pos2: np.ndarray, color, width):
        centerx, centery = tuple((pos1 + pos2) / 2)
        length = math.hypot(*(pos2 - pos1))
        angle = math.atan2(pos1[1] - pos2[1], pos1[0] - pos2[0])
        width2, length2 = width / 2, length / 2
        sin_ang, cos_ang = math.sin(angle), math.cos(angle)

        width2_sin_ang = width2 * sin_ang
        width2_cos_ang = width2 * cos_ang
        length2_sin_ang = length2 * sin_ang
        length2_cos_ang = length2 * cos_ang

        ul = (centerx + length2_cos_ang - width2_sin_ang,
              centery + width2_cos_ang + length2_sin_ang)
        ur = (centerx - length2_cos_ang - width2_sin_ang,
              centery + width2_cos_ang - length2_sin_ang)
        bl = (centerx + length2_cos_ang + width2_sin_ang,
              centery - width2_cos_ang + length2_sin_ang)
        br = (centerx - length2_cos_ang + width2_sin_ang,
              centery - width2_cos_ang - length2_sin_ang)

        gfxdraw.aapolygon(self.surface, (ul, ur, br, bl), color)
        gfxdraw.filled_polygon(self.surface, (ul, ur, br, bl), color)
