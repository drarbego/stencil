#!/usr/bin/env python3
"""
Creates a 3d printable stencil from an image.

Black pixels are cut from a plane in the X-Y axis, then the remaining
geometry is extruded in +Z — leaving a rectangular solid with the image
silhouette punched through it.

The plane is CANVAS_WIDTH × CANVAS_HEIGHT and the extrusion is 0.5 cm (Blender units = metres,
so 1 cm = 0.01 m).

Before processing, a hard contrast pass converts every pixel to pure black or
pure white based on whether its brightness is below or above 0.5.

Usage:
    blender --background --python silhouette.py -- <image_path> [--max_resolution N] [--depth D] [--normalized_width W] [--normalized_height H]

Arguments:
    image_path           Path to a black and white image (PNG, JPEG, etc.)
    --max_resolution     Maximum dimension in pixels before downscaling (default: 256)
    --depth              Extrusion distance in metres (default: 0.005 = 0.5 cm)
    --normalized_width   Fraction of canvas width for the mesh (default: 1.0)
    --normalized_height  Fraction of canvas height for the mesh (default: 1.0)

Example:
    --normalized_width 0.5 --normalized_height 0.5 on a 22×22 cm canvas → 11×11 cm mesh

Output:
    A .blend file written next to the source image.
"""

import argparse
import os
import sys

import bmesh
import bpy


DEFAULT_MAX_RESOLUTION = 256
DEFAULT_EXTRUDE_DEPTH = 0.005  # 0.5 cm in metres

# Physical canvas size in metres (22 cm × 22 cm)
# Adjust this to your printer's print bed dimensions
CANVAS_WIDTH = 0.22
CANVAS_HEIGHT = 0.22


def load_black_mask(image_path: str, max_res: int) -> tuple[list[list[bool]], int, int]:
    """
    Load an image and return a 2D boolean mask where True = black pixel.

    A hard-contrast pass is applied first: pixels with brightness < 0.5 become
    pure black, pixels >= 0.5 become pure white.  The image is then downscaled
    if either dimension exceeds max_res.
    """
    img = bpy.data.images.load(os.path.abspath(image_path))
    w, h = img.size

    scale = min(max_res / w, max_res / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if new_w != w or new_h != h:
        img.scale(new_w, new_h)
        w, h = new_w, new_h
        print(f"  Scaled image to {w}x{h}")

    # img.pixels is a flat RGBA array, origin at bottom-left
    pixels = list(img.pixels)
    bpy.data.images.remove(img)

    threshold = 0.5
    mask = []
    for y in range(h):
        row = []
        for x in range(w):
            idx = (y * w + x) * 4
            brightness = (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3
            # Hard contrast: snap to pure black or pure white before thresholding.
            row.append(brightness < threshold)
        mask.append(row)

    return mask, w, h


def build_silhouette(
    mask: list[list[bool]],
    width: int,
    height: int,
    extrude_depth: float,
    normalized_width: float = 1.0,
    normalized_height: float = 1.0,
) -> bpy.types.Object:
    """
    Build the extruded silhouette mesh from the pixel mask.

    The base face sits on the X-Y plane (Z=0).  White pixels become quad
    faces; black pixels are omitted.  The surviving geometry is extruded in
    +Z by extrude_depth, producing a solid with holes where the silhouette was.

    width_fraction / height_fraction scale the canvas dimensions so the mesh
    occupies only that fraction of CANVAS_WIDTH × CANVAS_HEIGHT.
    """
    mesh = bpy.data.meshes.new("Silhouette")
    obj = bpy.data.objects.new("Silhouette", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    # Effective canvas after applying normalized dimensions.
    eff_w = CANVAS_WIDTH * normalized_width
    eff_h = CANVAS_HEIGHT * normalized_height

    # Fit the image inside the effective canvas preserving aspect ratio.
    image_aspect = width / height
    canvas_aspect = eff_w / eff_h
    if image_aspect >= canvas_aspect:
        sx = eff_w / 2.0
        sy = (eff_w / image_aspect) / 2.0
    else:
        sx = (eff_h * image_aspect) / 2.0
        sy = eff_h / 2.0

    # Build a (width+1) × (height+1) vertex grid in the X-Y plane (Z=0).
    verts: dict[tuple[int, int], bmesh.types.BMVert] = {}
    for j in range(height + 1):
        for i in range(width + 1):
            x = (i / width - 0.5) * 2.0 * sx
            y = (j / height - 0.5) * 2.0 * sy
            verts[(i, j)] = bm.verts.new((x, y, 0.0))

    bm.verts.ensure_lookup_table()

    # Create one quad face per white (non-black) pixel.
    kept = 0
    for row_y in range(height):
        for col_x in range(width):
            if not mask[row_y][col_x]:
                bm.faces.new([
                    verts[(col_x,     row_y    )],
                    verts[(col_x + 1, row_y    )],
                    verts[(col_x + 1, row_y + 1)],
                    verts[(col_x,     row_y + 1)],
                ])
                kept += 1

    total = width * height
    print(f"  Faces kept: {kept}/{total}  ({total - kept} black pixels removed)")

    # Extrude the entire face region in +Z.  extrude_face_region handles
    # disconnected islands correctly, building walls around each hole.
    result = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    new_verts = [e for e in result["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(0.0, 0.0, extrude_depth), verts=new_verts)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return obj


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path")
    parser.add_argument("--max_resolution", type=int, default=DEFAULT_MAX_RESOLUTION)
    parser.add_argument("--depth", type=float, default=DEFAULT_EXTRUDE_DEPTH)
    parser.add_argument("--normalized_width", type=float, default=1.0)
    parser.add_argument("--normalized_height", type=float, default=1.0)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.image_path):
        print(f"Error: file not found: {args.image_path}")
        sys.exit(1)

    # Start with an empty scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    print(f"Loading: {args.image_path}")
    mask, width, height = load_black_mask(args.image_path, args.max_resolution)

    print(f"Building mesh ({width}×{height} grid)…")
    build_silhouette(mask, width, height, args.depth, args.normalized_width, args.normalized_height)

    base_path = os.path.splitext(os.path.abspath(args.image_path))[0]

    blend_path = base_path + ".blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved: {blend_path}")

    stl_path = base_path + ".stl"
    bpy.ops.export_mesh.stl(filepath=stl_path)
    print(f"Saved: {stl_path}")


main()
