import math
import sys
import time
from typing import Union

import numpy as np
import pygame
from pygame import gfxdraw

from constants import *
from graph import Graph

from tqdm import tqdm


class Visualizer:
    def __init__(self, drawing_lock):
        self.exit_flag = False

        self.graph = None
        self.drawing_lock = drawing_lock
        self.is_search_graph = False
        self.draw_necessary = False
        self.edge_colors = {}
        self.search_edge_colors = {}

        self.history_getter = None
        self.current_history = (0, 0)
        self.history = []
        self.history_mode = False
        self.history_mode_type = 0
        self.animation_time_progress = None
        self.redraw_edges = set()

        # setup pygame
        pygame.init()
        self.clock = pygame.time.Clock()

        # setup screen
        self.screen_size = SCREEN_SIZE
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        pygame.display.set_caption("Shortest Path Multicut")
        self.screen.fill(WHITE)
        self.surface = pygame.Surface(self.screen_size)
        self.surface.fill(COLOR_KEY)
        self.surface.set_colorkey(COLOR_KEY)

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
                    self.surface.set_colorkey(COLOR_KEY)
                    self.set_scale()
                    self.draw_necessary = True

            if self.graph is not None:
                if self.draw_necessary or self.history_mode:
                    if self.history_mode:
                        current_time = time.time()
                        while self.animation_time_progress < current_time:
                            self.set_edge_color_for_history()

                    self.surface.fill(COLOR_KEY)

                    with self.drawing_lock:
                        self.draw_graph()

                    if self.draw_necessary:
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
            self.calculate_edge_colors()

    def calculate_edge_colors(self):
        num_edges = len(self.graph.edges)
        cut_negative_color = PURPLE
        non_cut_negative_color = RED
        cut_positive_color = LIME
        non_cut_positive_color = GREEN

        if num_edges > 10:
            cut_negative_color = RED
            non_cut_negative_color = GREY
            cut_positive_color = RED
            non_cut_positive_color = None

        for edge in self.graph.edges(keys=True, data=True):
            cost = edge[3]["cost"]
            cut = edge[3]["id"] in self.multicut or (edge[1], edge[0], edge[2]) in self.multicut

            if cut:
                if cost > 0:
                    color = cut_positive_color
                else:
                    color = cut_negative_color
            else:
                if cost > 0:
                    color = non_cut_positive_color
                else:
                    color = non_cut_negative_color
            self.edge_colors[edge[3]["id"]] = color

    def set_multicut(self, multicut):
        self.multicut = multicut
        for edge_id in multicut:
            self.edge_colors[edge_id] = RED
        self.draw_necessary = True

    def set_history_file(self, filename, mode=0):
        self.history_mode_type = mode
        print("read history file")
        self.history = []
        with open(filename, "r") as file:
            lines = [line.strip() for line in file]
            for line in tqdm(lines):
                line_list = line.split(';')
                if len(line_list) != 3:
                    print("history file corrupted")
                    break
                search_history, cut, merge = line_list

                if len(search_history) > 0:
                    search_history = list(map(int, search_history.split(',')))
                else:
                    search_history = []
                if len(cut) > 0:
                    cut = list(map(int, cut.split(',')))
                else:
                    cut = []
                if len(merge) > 0:
                    merge = list(map(int, merge.split(',')))
                else:
                    merge = []

                self.history.append((search_history, cut, merge))

        self.history_mode = True
        # self.history_getter = history_getter
        self.animation_time_progress = time.time()

    def get_history(self):
        # if self.history_getter is None:
        #     return None
        # return self.history_getter()
        return self.history

    def set_edge_color_for_history(self):
        self.animation_time_progress += 0.01
        history = self.get_history()
        if len(history) == 0:
            return
        major_history = self.current_history[0]
        minor_history = self.current_history[1]
        current_history_state = history[major_history]
        if self.history_mode_type != 0 and len(current_history_state[0]) == 0 and len(current_history_state[1]) == 0 and len(current_history_state[2]) == 0:
            self.search_edge_colors = {}
            self.draw_necessary = True

        minor_history += 1
        if minor_history > len(current_history_state[0]):
            if self.history_mode_type == 0:
                self.search_edge_colors = {}
            self.draw_necessary = True
            if major_history == len(history) - 1:
                return
            major_history += 1
            minor_history = 0

        self.current_history = (major_history, minor_history)

        if minor_history == 0 and major_history > 0:
            current_history_state = history[major_history - 1]

            for edge in current_history_state[1]:
                # cuts
                self.edge_colors[edge] = RED
            for edge in current_history_state[2]:
                # paths
                self.edge_colors[edge] = DARK_GREY
        else:
            edge = current_history_state[0][minor_history - 1]
            self.search_edge_colors[edge] = GREEN
            self.redraw_edges.add(edge)

    def get_edge_color(self, edge_id):
        color = self.search_edge_colors.get(edge_id, None)
        if color is None:
            color = self.edge_colors[edge_id]
        return color

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
            if not self.draw_necessary and edge[3]["id"] not in self.redraw_edges:
                continue

            pos1 = self.graph.nodes[edge[0]]['pos']
            pos2 = self.graph.nodes[edge[1]]['pos']
            if pos2[0] < 0 and pos2[1] < 0:
                positions = self.get_outside_pos(pos1, max_x, max_y)
                pos2 = positions[0]
                if tuple(pos2) in outside_positions:
                    pos2 = positions[1]
                outside_positions.append(tuple(pos2))

            color = self.get_edge_color(edge[3]["id"])

            if color is not None:
                if num_edges < 1000:
                    self.draw_thick_aaline(self.scale_value(pos1 + pos_offset[0]),
                                           self.scale_value(pos2 + pos_offset[1]),
                                           color,
                                           edge_width)
                else:
                    pygame.draw.line(self.surface, color,
                                     self.scale_value(pos1 + pos_offset[0]),
                                     self.scale_value(pos2 + pos_offset[1]))

        self.redraw_edges = set()

        if not self.draw_necessary:
            return

        if node_radius < 5:
            return
        # draw nodes
        outside_node = 0
        for node, data in self.graph.nodes(data=True):
            if data["pos"][0] < 0 and data["pos"][1]:
                outside_node = node
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
            data = self.graph.nodes[outside_node]
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
