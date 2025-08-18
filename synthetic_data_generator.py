import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import numpy as np
import matplotlib.pyplot as plt
import random, os, json, cv2, math
from PIL import Image
import bpy
import colorsys

cam_path   = "/datashare/project/camera.json"
bg_dir     = "/datashare/project/train2017"
num_images = 48


hdr_base_dir = "/datashare/project/haven/hdris"

hdr_files = []
for root, dirs, files in os.walk(hdr_base_dir):
    for file in files:
        if file.endswith('.hdr'):
            hdr_files.append(os.path.join(root, file))

# Choosing a random HDR file from the list
random_hdr_file = random.choice(hdr_files)

# Set the world background using the randomly chosen HDRI
bproc.world.set_world_background_hdr_img(random_hdr_file)

bproc.init()

# post-process helpers 
def random_motion_blur(img_arr, max_kernel=15):
    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k))
    direction = random.choice(["h", "v", "d"])
    if direction == "h":
        kernel[k // 2, :] = np.ones(k)
    elif direction == "v":
        kernel[:, k // 2] = np.ones(k)
    else:
        np.fill_diagonal(kernel, 1)
    kernel = kernel / kernel.sum()
    img_blur = cv2.filter2D(img_arr, -1, kernel)
    return img_blur

def _bbox_and_axis_from_uv(uv, names=None):
    """Return bbox center, scale (diag), and main axis angle (deg) from 2D keypoints."""
    uv = np.asarray(uv, float)
    min_xy = np.nanmin(uv, axis=0)
    max_xy = np.nanmax(uv, axis=0)
    center = (min_xy + max_xy) / 2.0
    diag = float(np.linalg.norm(max_xy - min_xy))
    angle_deg = 0.0

    if names is not None:
        name2idx = {n: i for i, n in enumerate(names)}
        pairs = [
            ("connector", "tip_l"), ("connector", "tip_r"),
            ("handle_l", "handle_r"), ("tip_b", "tip_l"), ("tip_b", "tip_r")
        ]
        for a, b in pairs:
            if a in name2idx and b in name2idx:
                pa, pb = uv[name2idx[a]], uv[name2idx[b]]
                v = pb - pa
                if np.linalg.norm(v) > 1e-3:
                    angle_deg = math.degrees(math.atan2(v[1], v[0]))
                    break
    else:
        v = max_xy - min_xy
        if np.linalg.norm(v) > 1e-3:
            angle_deg = math.degrees(math.atan2(v[1], v[0]))

    return center, diag, angle_deg

def add_glove_connected_occlusion(img_bgr, uv, img_w, img_h,
                                  kp_names=None,
                                  prob=0.8,
                                  segments=3,                 # number of lobes in the chain
                                  overlap=0.5,                # fraction of major that overlaps (0..0.95)
                                  major_rel=(0.22, 0.32),     # long axis (same for all lobes) as fraction of diag
                                  minor_rel=(0.55, 0.75),     # short/long axis ratio (same for all lobes)
                                  rot_jitter=3,               # very small angular noise (deg)
                                  perp_jitter_rel=0.005,      # tiny perpendicular jitter (fraction of diag)
                                  color_rng=((180,235,235),(220,255,255)),  # BGR white-yellow
                                  dilate_px=2,                # stitch borders to avoid 1px seams
                                  return_mask=False):
    """
    Single connected, opaque glove-like shape constructed from equal-size, overlapping ellipses.
    The spacing is chosen so that adjacent ellipses overlap by `overlap * major` along the tool axis.
    """
    if random.random() > prob:
        return (img_bgr, np.zeros((img_h, img_w), np.float32)) if return_mask else img_bgr

    img = img_bgr.copy()
    center, diag, axis_deg = _bbox_and_axis_from_uv(uv, kp_names)
    if not np.isfinite(diag) or diag < 10:
        diag = max(img_w, img_h) * 0.25

    # tool axis unit vectors
    a_hat = np.array([math.cos(math.radians(axis_deg)),
                      math.sin(math.radians(axis_deg))], np.float32)
    b_hat = np.array([-a_hat[1], a_hat[0]], np.float32)

    # color
    (b0,g0,r0), (b1,g1,r1) = color_rng
    color = (random.randint(b0,b1), random.randint(g0,g1), random.randint(r0,r1))

    # one size for all lobes to control overlap deterministically
    major = random.uniform(*major_rel) * diag
    minor = random.uniform(minor_rel[0]*major, minor_rel[1]*major)

    # center-to-center step that guarantees overlap
    step = (1.0 - max(0.0, min(overlap, 0.95))) * major

    # centers arranged symmetrically around the tool center
    idxs = np.arange(segments, dtype=np.float32) - (segments - 1) / 2.0
    mask = np.zeros((img_h, img_w), np.uint8)

    # a single small global angle jitter keeps all lobes consistently oriented
    angle = axis_deg + random.uniform(-rot_jitter, rot_jitter)

    for i in idxs:
        base = center + (i * step) * a_hat
        c = base + (random.uniform(-perp_jitter_rel, perp_jitter_rel) * diag) * b_hat
        cx = int(np.clip(c[0], 0, img_w - 1))
        cy = int(np.clip(c[1], 0, img_h - 1))

        cv2.ellipse(mask, (cx, cy), (int(major/2), int(minor/2)),
                    angle, 0, 360, 1, -1, lineType=cv2.LINE_8)

    if dilate_px > 0:
        k = int(max(1, round(dilate_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    img[mask == 1] = color
    return (img, mask.astype(np.float32)) if return_mask else img
# -----------------------------------------------------------
surgical_tools = ['needle_holder', 'tweezers']
enabled = False

for tool in surgical_tools:
    if tool == 'needle_holder':
        obj_path = "/datashare/project/surgical_tools_models/needle_holder"
        # obj_list = [f'{obj_path}/NH{i}.obj' for i in range(1, 16)]
        obj_list = [f'{obj_path}/NH{i}.obj' for i in range(1, 3)]

        output_dir = "/home/student/project_2D/kp_nh2"
        kp_json    = "/home/student/project_2D/keypoints_needle.json"
        offset_u = 3
        offset_v = 20

    elif tool == 'tweezers':
        obj_path = "/datashare/project/surgical_tools_models/tweezers"
        # obj_list = [f'{obj_path}/T{i}.obj' for i in range(1, 11)]
        obj_list = [f'{obj_path}/T{i}.obj' for i in range(1, 3)]

        output_dir = "/home/student/project_2D/kp_tw2"
        kp_json    = "/home/student/project_2D/keypoints_tweezers.json"
        offset_u = 0
        offset_v = 20

    for obj in obj_list:

        bproc.clean_up()
        current_obj = obj

        model_name = os.path.splitext(os.path.basename(current_obj))[0]
        print(f"Processing {model_name}...\n\n")
        model_output_dir = os.path.join(output_dir, model_name)
        rgb_dir, depth_dir, ann_dir = [os.path.join(model_output_dir, d) for d in ("RGB", "Depth", "Annotations")]
        for d in (rgb_dir, depth_dir, ann_dir):
            os.makedirs(d, exist_ok=True)

        with open(kp_json) as f:
            kp_local_dict = json.load(f)[model_name]
        kp_names = list(kp_local_dict.keys())
        kp_local = np.array([kp_local_dict[n] for n in kp_names], dtype=np.float32)

        # Initialize scene and load object
        obj = bproc.loader.load_obj(current_obj)[0]
        obj.set_cp("category_id", 1)

        # Materials
        if tool == 'needle_holder':
            mat = obj.get_materials()[0]  # metal
            mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
            mat.set_principled_shader_value("Metallic", 1)
            mat.set_principled_shader_value("Roughness", 0.2)

            mat = obj.get_materials()[1]  # gold tint
            h = np.random.uniform(0.10, 0.14)
            s = np.random.uniform(0.85, 1.0)
            v = np.random.uniform(0.8, 1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            gold_color = [r, g, b, 1.0]
            mat.set_principled_shader_value("Base Color", gold_color)
            mat.set_principled_shader_value("Roughness", random.uniform(0, 1))
            mat.set_principled_shader_value("Metallic", 1)
        else:
            for mat in obj.get_materials():
                mat.set_principled_shader_value("Metallic", 1.0)
                mat.set_principled_shader_value("Roughness", 0.2)

        # Local -> world for keypoints
        M = np.asarray(obj.blender_obj.matrix_world, dtype=np.float32)
        kp_local_h = np.concatenate([kp_local, np.ones((kp_local.shape[0], 1))], 1)
        kp_world = (M @ kp_local_h.T).T[:, :3]

        # Camera intrinsics
        with open(cam_path) as f:
            cam = json.load(f)
        K = np.array([[cam["fx"], 0, cam["cx"]],
                      [0, cam["fy"], cam["cy"]],
                      [0, 0, 1]], dtype=np.float32)
        w, h = cam["width"], cam["height"]
        CameraUtility.set_intrinsics_from_K_matrix(K, w, h)

        # Renderer outputs
        if not enabled:
            bproc.renderer.enable_depth_output(True)
            enabled = True

        bproc.renderer.set_output_format(enable_transparency=False)
        bproc.renderer.enable_segmentation_output(map_by=["instance"])

        # Background images
        bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Light
        light = bproc.types.Light()
        light.set_type("POINT")

        # Rendering loop
        poi = obj.get_location()

        for i in range(num_images):
            # Camera pose
            loc = bproc.sampler.shell(poi, 5, 8, -90, 90, -180, 180)
            rot = bproc.camera.rotation_from_forward_vec(
                poi - loc, inplane_rot=random.uniform(-math.pi, math.pi))
            cam2world = bproc.math.build_transformation_mat(loc, rot)
            bproc.camera.add_camera_pose(cam2world, frame=0)

            # Per-frame area light
            light = bproc.types.Light()
            light.set_type("AREA")
            light.set_energy(random.uniform(500, 1000))
            light.set_color([
                random.uniform(0.95, 1.0),
                random.uniform(0.9, 0.97),
                random.uniform(0.85, 0.95)
            ])
            light.set_location(poi + np.array([
                random.uniform(-3, 3),
                random.uniform(-3, 3),
                random.uniform(6, 10)
            ]))

            # Render
            bpy.context.scene.frame_set(0)
            data = bproc.renderer.render()

            # RGBA composite via segmentation alpha
            rgb_f = (data["colors"][0] * 255).astype(np.uint8)[..., :3]
            seg = data["instance_segmaps"][0]
            alpha = (seg > 0).astype(np.uint8) * 255
            rgba = np.dstack([rgb_f, alpha])

            comp = Image.alpha_composite(
                Image.open(random.choice(bg_files)).resize((w, h)).convert("RGBA"),
                Image.fromarray(rgba, "RGBA"))
            rgb = cv2.cvtColor(np.array(comp), cv2.COLOR_RGBA2BGR)

            # Adding motion blur
            if random.random() < 0.3:
                rgb = random_motion_blur(rgb)

            # Depth
            depth = data["depth"][0].squeeze()

            # Project 3D keypoints to 2D
            uv = bproc.camera.project_points(kp_local, frame=0)

            # World->camera for KP depth
            W2C = np.linalg.inv(cam2world)
            kp_w_h = np.concatenate([kp_world, np.ones((kp_world.shape[0], 1))], 1).T
            kp_cam = W2C @ kp_w_h
            kp_z = kp_cam[2]  # negative = in front of cam

            # 1) Build annotations 
            ann = {}
            for idx, name in enumerate(kp_names):
                u, v = map(int, uv[idx])
                z_cam = kp_z[idx]
                visible_geom = (z_cam < 0) and (0 <= u < w) and (0 <= v < h)
                ann[name] = {"pixel": [u, v], "depth": float(-z_cam), "visible": bool(visible_geom)}

            # 2) Add one connected, OPAQUE glove occlusion and get binary mask
            rgb, occ_mask = add_glove_connected_occlusion(
                rgb, uv, w, h,
                kp_names=kp_names,
                prob=0.8,
                segments=3,
                overlap=0.5,
                rot_jitter=3,
                perp_jitter_rel=0.005,
                major_rel=(0.22, 0.32),
                minor_rel=(0.55, 0.75),
                dilate_px=2,
                return_mask=True
            )

            # 3) Conservative coverage test
            kp_radius = 5
            inner_margin_px = 1      # 
            kern_size = 2 * (kp_radius + inner_margin_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))

            occ_mask_bin = (occ_mask >= 0.5).astype(np.uint8)
            occ_mask_strict = cv2.erode(occ_mask_bin, kernel, iterations=1)

            # 4) Update visibility
            for idx, name in enumerate(kp_names):
                u, v = ann[name]["pixel"]
                px, py = u + offset_u, v + offset_v  # where the label is actually drawn
                if 0 <= px < w and 0 <= py < h and occ_mask_strict[py, px] == 1:
                    ann[name]["visible"] = False


            # 5) Draw ONLY visible keypoints/labels
            cmap = plt.get_cmap('tab10')
            vis = 0
            for idx, name in enumerate(kp_names):
                if not ann[name]["visible"]:
                    continue
                u, v = ann[name]["pixel"]
                color_rgb = cmap(idx % 10)[:3]
                color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])

                px, py = u + offset_u, v + offset_v
                cv2.circle(rgb, (px, py), kp_radius, color_bgr, -1)
                cv2.circle(rgb, (px, py), kp_radius, (0, 0, 0), 1)
                label_pos = (px + 6, py + 12)
                cv2.putText(rgb, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 2, cv2.LINE_AA)
                cv2.putText(rgb, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                vis += 1

            # Save
            cv2.imwrite(f"{rgb_dir}/img_{i:04d}.png", rgb)
            cv2.imwrite(f"{depth_dir}/depth_{i:04d}.png", (depth * 1000).astype(np.uint16))
            with open(f"{ann_dir}/ann_{i:04d}.json", "w") as f:
                json.dump({
                    "K": K.astype(float).tolist(),
                    "cam2world": np.asarray(cam2world, float).tolist(),
                    "keypoints": ann
                }, f, indent=2)

            print(f"Saved image {i} ({vis} kp visible)")
