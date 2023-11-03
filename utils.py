import colorsys


def generate_distinct_colors(n):
    colors = []
    hue_values = [i / n for i in range(n)]
    saturation = 0.5
    lightness = 0.5

    for hue in hue_values:
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        colors.append((r, g, b))

    return colors
