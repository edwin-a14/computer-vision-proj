#!/usr/bin/env python3
"""
Interactive Matplotlib-based color mask adjuster for histogram validation.

USAGE:
    python src/sample_pixels.py <image_path> [--color white|red|yellow|blue|orange|black|green]

INTERACTIVE CONTROLS:
    LEFT-CLICK on pixels to sample their HSV values
    DRAG sliders to adjust mask thresholds in real-time
    KEYBOARD:
        'u' = Undo last sample
        'r' = Reset all samples and sliders
        'q' = Quit

WORKFLOW:
    1. Open image with stop signs that have missing white pixels
    2. Click on white pixels that SHOULD be captured but aren't (show as gray in mask viz)
    3. Drag sliders to adjust thresholds until those pixels turn green
    4. Adjust until satisfied
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import argparse

# Try importing color mask logic for reference
try:
    from color_shape_prep import build_color_masks
except ImportError:
    build_color_masks = None
try:
    from utils import apply_gray_world
except ImportError:
    apply_gray_world = None


class ColorMaskAdjuster:
    """Interactive tool for sampling pixels and adjusting color mask thresholds."""

    def _get_selection_ranges_and_hist(self):
        """Return (h_vals, s_vals, v_vals, l_vals, sel_h, sel_s, sel_v, sel_l) for current selection or mask-accepted pixels if none."""
        if self.sampled_pixels:
            h_vals = [p[0] for p in self.sampled_pixels]
            s_vals = [p[1] for p in self.sampled_pixels]
            v_vals = [p[2] for p in self.sampled_pixels]
            hls_vals = [tuple(self.img_hls[y, x]) for (x, y) in self.sampled_coords]
            l_vals = [int(p[1]) for p in hls_vals]
        else:
            mask = None
            if build_color_masks is not None:
                try:
                    hsv = self.img_hsv
                    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
                    l_hls = self.img_hls[:,:,1]
                    masks = build_color_masks(h, s, v, l_hls, self.img_bgr)
                    key_map = {
                        'white': 'white_mask',
                        'red': 'red_mask',
                        'yellow': 'yellow_mask',
                        'blue': 'blue_mask',
                        'orange': 'orange_mask',
                        'black': 'black_mask',
                        'green': 'green_mask',
                    }
                    color_key = key_map.get(self.target_color)
                    if color_key and color_key in masks:
                        mask = masks[color_key]
                except Exception:
                    mask = None
            if mask is None:
                mask = self._build_current_mask()
            mask_indices = np.where(mask > 0)
            if mask_indices[0].size > 0:
                h_vals = self.img_hsv[:,:,0][mask_indices].tolist()
                s_vals = self.img_hsv[:,:,1][mask_indices].tolist()
                v_vals = self.img_hsv[:,:,2][mask_indices].tolist()
                l_vals = self.img_hls[:,:,1][mask_indices].tolist()
            else:
                h_vals = s_vals = v_vals = l_vals = []
        if h_vals:
            sel_h = f"{min(h_vals)}-{max(h_vals)}"
        else:
            sel_h = "-"
        if s_vals:
            sel_s = f"{min(s_vals)}-{max(s_vals)}"
        else:
            sel_s = "-"
        if v_vals:
            sel_v = f"{min(v_vals)}-{max(v_vals)}"
        else:
            sel_v = "-"
        if l_vals:
            sel_l = f"{min(l_vals)}-{max(l_vals)}"
        else:
            sel_l = "-"
        return h_vals, s_vals, v_vals, l_vals, sel_h, sel_s, sel_v, sel_l

    def __init__(self, image_path, target_color='white', skip_wb=False):
        """
        Parameters:
        -----------
        image_path : str
            Path to image file
        target_color : str
            Color to adjust: white, red, yellow, blue, orange, black, green
        """
        self.image_path = image_path
        self.target_color = target_color.lower()
        
        # Load image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Optionally skip white balance
        if skip_wb or apply_gray_world is None:
            img_bgr_wb = img_bgr
        else:
            try:
                img_bgr_wb = apply_gray_world(img_bgr)
            except Exception:
                img_bgr_wb = img_bgr

        self.img_bgr = img_bgr_wb
        self.img_rgb = cv2.cvtColor(img_bgr_wb, cv2.COLOR_BGR2RGB)
        self.img_hsv = cv2.cvtColor(img_bgr_wb, cv2.COLOR_BGR2HSV)
        self.img_hls = cv2.cvtColor(img_bgr_wb, cv2.COLOR_BGR2HLS)
        
        # Sampled pixels (HSV values)
        self.sampled_pixels = []
        self.sampled_coords = []
        
        # Current threshold values (will be populated by default thresholds)
        self.thresholds = self._get_default_thresholds()
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Color Mask Adjuster - {self.target_color.upper()} | {Path(image_path).name}')
        
        # Create subplots: 2x3 grid
        # Top row: image with selections, mask visualization, HSV histogram
        # Bottom row: HLS histogram, stats, sliders
        self.ax_image = plt.subplot(2, 3, 1)
        self.ax_mask = plt.subplot(2, 3, 2)
        self.ax_hsv_hist = plt.subplot(2, 3, 3)
        self.ax_hls_hist = plt.subplot(2, 3, 4)
        self.ax_stats = plt.subplot(2, 3, 5)
        self.ax_sliders = plt.subplot(2, 3, 6)
        self.ax_sliders.axis('off')
        
        # Store image display objects
        self.im_image = None
        self.im_mask = None
        
        # Track zoom state per axis
        self.zoom_state = {
            'ax_image': None,
            'ax_mask': None,
        }
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Slider objects (created in setup_sliders)
        self.sliders = {}
        self.slider_initialized = False
        
        # Draw initial state
        self.update_display()
    
    def _get_default_thresholds(self):
        """Return full range for all channels as default thresholds."""
        return {
            'h_min': 0, 'h_max': 180,
            's_min': 0, 's_max': 255,
            'v_min': 0, 'v_max': 255,
            'l_min': 0, 'l_max': 255,
        }
    
    def _on_click(self, event):
        """Handle mouse clicks to sample pixels."""
        if event.inaxes != self.ax_image or event.xdata is None or event.ydata is None:
            return
        
        # Use data coordinates (already handles zoom correctly)
        x, y = int(round(event.xdata)), int(round(event.ydata))
        
        # Bounds check
        h, w = self.img_hsv.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            hsv_val = tuple(self.img_hsv[y, x])
            hls_val = tuple(self.img_hls[y, x])
            # Convert numpy types to Python ints for clean output
            hsv_clean = (int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2]))
            l_val = int(hls_val[1])
            self.sampled_pixels.append(hsv_clean)
            self.sampled_coords.append((x, y))
            print(f"Sampled pixel at ({x}, {y}): HSV={hsv_clean}, L={l_val}")
            self.update_display_partial()  # Partial update to preserve zoom
    
    def _on_scroll(self, event):
        """Handle mouse wheel zoom."""
        if event.inaxes not in [self.ax_image, self.ax_mask]:
            return
        
        cur_xlim = event.inaxes.get_xlim()
        cur_ylim = event.inaxes.get_ylim()
        
        xdata, ydata = event.xdata, event.ydata
        
        # Zoom factor
        if event.button == 'up':
            scale_factor = 0.8  # Zoom in
        elif event.button == 'down':
            scale_factor = 1.2  # Zoom out
        else:
            return
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        event.inaxes.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        event.inaxes.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        # Store zoom state
        if event.inaxes == self.ax_image:
            self.zoom_state['ax_image'] = (event.inaxes.get_xlim(), event.inaxes.get_ylim())
        else:
            self.zoom_state['ax_mask'] = (event.inaxes.get_xlim(), event.inaxes.get_ylim())
        
        self.fig.canvas.draw_idle()
    
    def _on_key(self, event):
        """Handle keyboard commands (u=Undo, r=Reset, q=Quit)."""
        if event.key == 'u':
            if self.sampled_pixels:
                self.sampled_pixels.pop()
                self.sampled_coords.pop()
                print("Undid last sample")
                self.update_display_partial()
        elif event.key == 'r':
            self.sampled_pixels.clear()
            self.sampled_coords.clear()
            self.thresholds = self._get_default_thresholds()
            self.zoom_state = {'ax_image': None, 'ax_mask': None}
            print("Reset samples and thresholds")
            self.update_display()
        elif event.key == 'q':
            plt.close(self.fig)
    
    
    

    
    def update_display(self):
        """Update all plot panels."""
        self.ax_image.clear()
        self.ax_mask.clear()
        self.ax_hsv_hist.clear()
        self.ax_hls_hist.clear()
        self.ax_stats.clear()
        
        # 1. Show image with sample points
        self.im_image = self.ax_image.imshow(self.img_rgb)
        for i, (x, y) in enumerate(self.sampled_coords):
            self.ax_image.plot(x, y, 'g+', markersize=10, markeredgewidth=2)
        self.ax_image.set_title(f'Image with Samples ({len(self.sampled_pixels)})')
        self.ax_image.set_xlim(0, self.img_rgb.shape[1])
        self.ax_image.set_ylim(self.img_rgb.shape[0], 0)
        
        # 2. Show mask visualization
        # Prefer using repo's build_color_masks when available for fidelity
        mask = None
        if build_color_masks is not None:
            try:
                hsv = self.img_hsv
                h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
                l_hls = self.img_hls[:,:,1]
                masks = build_color_masks(h, s, v, l_hls, self.img_bgr)
                key_map = {
                    'white': 'white_mask',
                    'red': 'red_mask',
                    'yellow': 'yellow_mask',
                    'blue': 'blue_mask',
                    'orange': 'orange_mask',
                    'black': 'black_mask',
                    'green': 'green_mask',
                }
                color_key = key_map.get(self.target_color)
                if color_key and color_key in masks:
                    mask = masks[color_key]
            except Exception:
                mask = None
        if mask is None:
            mask = self._build_current_mask()
        mask_display = np.zeros_like(self.img_rgb)
        mask_display[mask > 0] = [0, 255, 0]  # Green where mask is true
        mask_display[mask == 0] = [200, 200, 200]  # Gray where mask is false
        
        self.im_mask = self.ax_mask.imshow(mask_display)
        # Keep the title simple; show code quote and summary in stats panel
        self.ax_mask.set_title('Mask Visualization (Green=Captured, Gray=Missing)')
        self.ax_mask.set_xlim(0, self.img_rgb.shape[1])
        self.ax_mask.set_ylim(self.img_rgb.shape[0], 0)
        
        # 3. HSV/HLS histogram and selection ranges (DRY)
        h_vals, s_vals, v_vals, l_vals, sel_h, sel_s, sel_v, sel_l = self._get_selection_ranges_and_hist()
        if self.sampled_pixels:
            title_hsv = f'HSV Mean (n={len(self.sampled_pixels)})'
            title_hls = f'HLS Mean (n={len(self.sampled_pixels)})'
        else:
            title_hsv = 'HSV Mean (mask accepted)'
            title_hls = 'HLS Mean (mask accepted)'
        # HSV histogram
        if h_vals and s_vals and v_vals:
            self.ax_hsv_hist.bar(['H', 'S', 'V'], 
                                 [np.mean(h_vals), np.mean(s_vals), np.mean(v_vals)],
                                 color=['red', 'green', 'blue'], alpha=0.7)
            self.ax_hsv_hist.set_ylim(0, 255)
            self.ax_hsv_hist.set_title(title_hsv)
            self.ax_hsv_hist.set_ylabel('Value')
            h_range = f"{min(h_vals)}-{max(h_vals)}"
            s_range = f"{min(s_vals)}-{max(s_vals)}"
            v_range = f"{min(v_vals)}-{max(v_vals)}"
            self.ax_hsv_hist.text(0.02, 0.98, f"Ranges:\nH: {h_range}\nS: {s_range}\nV: {v_range}",
                                 transform=self.ax_hsv_hist.transAxes,
                                 verticalalignment='top', fontsize=8, family='monospace')
        else:
            self.ax_hsv_hist.set_title(title_hsv)
        # HLS histogram
        if l_vals:
            # For HLS, get H, L, S from accepted pixels
            if self.sampled_pixels:
                hls_vals = [tuple(self.img_hls[y, x]) for (x, y) in self.sampled_coords]
                h_hls = [int(p[0]) for p in hls_vals]
                l_hls = [int(p[1]) for p in hls_vals]
                s_hls = [int(p[2]) for p in hls_vals]
            else:
                mask = None
                if build_color_masks is not None:
                    try:
                        hsv = self.img_hsv
                        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
                        l_hls_img = self.img_hls[:,:,1]
                        masks = build_color_masks(h, s, v, l_hls_img, self.img_bgr)
                        key_map = {
                            'white': 'white_mask',
                            'red': 'red_mask',
                            'yellow': 'yellow_mask',
                            'blue': 'blue_mask',
                            'orange': 'orange_mask',
                            'black': 'black_mask',
                            'green': 'green_mask',
                        }
                        color_key = key_map.get(self.target_color)
                        if color_key and color_key in masks:
                            mask = masks[color_key]
                    except Exception:
                        mask = None
                if mask is None:
                    mask = self._build_current_mask()
                mask_indices = np.where(mask > 0)
                hls_vals = [tuple(self.img_hls[y, x]) for y, x in zip(*mask_indices)]
                h_hls = [int(p[0]) for p in hls_vals]
                l_hls = [int(p[1]) for p in hls_vals]
                s_hls = [int(p[2]) for p in hls_vals]
            if h_hls and l_hls and s_hls:
                self.ax_hls_hist.bar(['H', 'L', 'S'], [np.mean(h_hls), np.mean(l_hls), np.mean(s_hls)],
                                     color=['orange', 'yellow', 'cyan'], alpha=0.7)
                self.ax_hls_hist.set_ylim(0, 255)
                self.ax_hls_hist.set_title(title_hls)
                self.ax_hls_hist.set_ylabel('Value')
                h_range = f"{min(h_hls)}-{max(h_hls)}"
                l_range = f"{min(l_hls)}-{max(l_hls)}"
                s_range = f"{min(s_hls)}-{max(s_hls)}"
                self.ax_hls_hist.text(0.02, 0.98, f"Ranges:\nH: {h_range}\nL: {l_range}\nS: {s_range}",
                                     transform=self.ax_hls_hist.transAxes,
                                     verticalalignment='top', fontsize=8, family='monospace')
            else:
                self.ax_hls_hist.set_title(title_hls)
        else:
            self.ax_hls_hist.set_title(title_hls)
        
        # 4. Show current thresholds as text (plus quoted code, wrapped and centered)
        thresh = self.thresholds
        stats_text = f"""Current Thresholds:
        
H: {thresh['h_min']}-{thresh['h_max']}
S: {thresh['s_min']}-{thresh['s_max']}
V: {thresh['v_min']}-{thresh['v_max']}
"""
        if 'l_min' in thresh:
            stats_text += f"L: {thresh['l_min']}-{thresh['l_max']}\n"
        code_line = self._get_mask_code_snippet(self.target_color)
        if code_line:
            wrapped = self._wrap_text(code_line, width=54)
            stats_text += "\nQuoted Mask Code (from color_shape_prep.py):\n" + wrapped + "\n"
        
        stats_text += f"""
    Samples: {len(self.sampled_pixels)}

    KEYBOARD:
    u = Undo
    r = Reset
    q = Quit

    MOUSE:
    Scroll = Zoom
    Click = Sample

    SLIDERS:
    Drag to adjust thresholds
    """
        # Centered stats text so quoted code is centered on initial render
        self.ax_stats.text(0.5, 0.97, stats_text, transform=self.ax_stats.transAxes,
                  verticalalignment='top', horizontalalignment='center',
                  fontfamily='monospace', fontsize=7)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Stats & Controls')
        
        # Create sliders only once
        if not self.slider_initialized:
            self._create_sliders()
            self.slider_initialized = True
        
        # Restore zoom state if it exists
        if self.zoom_state['ax_image']:
            xlim, ylim = self.zoom_state['ax_image']
            self.ax_image.set_xlim(xlim)
            self.ax_image.set_ylim(ylim)
        
        if self.zoom_state['ax_mask']:
            xlim, ylim = self.zoom_state['ax_mask']
            self.ax_mask.set_xlim(xlim)
            self.ax_mask.set_ylim(ylim)
        
        # Avoid tight_layout due to incompatible Axes causing warnings/overlap
        # plt.tight_layout()
        self.fig.canvas.draw_idle()
    
    def _create_sliders(self):
        """Create interactive sliders for threshold adjustment."""
        thresh = self.thresholds
        
        # Clear the slider panel
        self.ax_sliders.clear()
        self.ax_sliders.axis('off')
        
        # Create a dedicated slider area (use figure instead of axes)
        slider_height = 0.35
        slider_y_start = 0.25
        
        # H range sliders
        ax_h_min = self.fig.add_axes([0.72, slider_y_start, 0.15, 0.02])
        ax_h_max = self.fig.add_axes([0.72, slider_y_start - 0.03, 0.15, 0.02])
        
        # S range sliders
        ax_s_min = self.fig.add_axes([0.72, slider_y_start - 0.06, 0.15, 0.02])
        ax_s_max = self.fig.add_axes([0.72, slider_y_start - 0.09, 0.15, 0.02])
        
        # V range sliders
        ax_v_min = self.fig.add_axes([0.72, slider_y_start - 0.12, 0.15, 0.02])
        ax_v_max = self.fig.add_axes([0.72, slider_y_start - 0.15, 0.15, 0.02])
        # L range sliders (always present)
        has_l = 'l_min' in thresh
        ax_l_min = self.fig.add_axes([0.72, slider_y_start - 0.18, 0.15, 0.02])
        ax_l_max = self.fig.add_axes([0.72, slider_y_start - 0.21, 0.15, 0.02])
        
        # Create sliders
        slider_h_min = Slider(ax_h_min, 'H min', 0, 180, valinit=thresh['h_min'], valstep=1)
        slider_h_max = Slider(ax_h_max, 'H max', 0, 180, valinit=thresh['h_max'], valstep=1)
        slider_s_min = Slider(ax_s_min, 'S min', 0, 255, valinit=thresh['s_min'], valstep=1)
        slider_s_max = Slider(ax_s_max, 'S max', 0, 255, valinit=thresh['s_max'], valstep=1)
        slider_v_min = Slider(ax_v_min, 'V min', 0, 255, valinit=thresh['v_min'], valstep=1)
        slider_v_max = Slider(ax_v_max, 'V max', 0, 255, valinit=thresh['v_max'], valstep=1)
        slider_l_min = Slider(ax_l_min, 'L min', 0, 255, valinit=thresh['l_min'], valstep=1)
        slider_l_max = Slider(ax_l_max, 'L max', 0, 255, valinit=thresh['l_max'], valstep=1)
        
        # Update callback
        def on_slider_change(val):
            self.thresholds['h_min'] = int(slider_h_min.val)
            self.thresholds['h_max'] = int(slider_h_max.val)
            self.thresholds['s_min'] = int(slider_s_min.val)
            self.thresholds['s_max'] = int(slider_s_max.val)
            self.thresholds['v_min'] = int(slider_v_min.val)
            self.thresholds['v_max'] = int(slider_v_max.val)
            if has_l:
                self.thresholds['l_min'] = int(slider_l_min.val)
                self.thresholds['l_max'] = int(slider_l_max.val)
            
            # Preserve zoom state
            mask_xlim = self.ax_mask.get_xlim()
            mask_ylim = self.ax_mask.get_ylim()
            
            # Update mask visualization
            self.ax_mask.clear()
            mask = self._build_current_mask()
            mask_display = np.zeros_like(self.img_rgb)
            mask_display[mask > 0] = [0, 255, 0]
            mask_display[mask == 0] = [200, 200, 200]
            self.im_mask = self.ax_mask.imshow(mask_display)
            self.ax_mask.set_title('Mask Visualization (Green=Captured, Gray=Missing)')
            self.ax_mask.set_xlim(mask_xlim)
            self.ax_mask.set_ylim(mask_ylim)
            
            # Update stats with full info
            self.ax_stats.clear()
            stats_text = f"""Current Thresholds:

H: {self.thresholds['h_min']}-{self.thresholds['h_max']}
S: {self.thresholds['s_min']}-{self.thresholds['s_max']}
V: {self.thresholds['v_min']}-{self.thresholds['v_max']}

Samples: {len(self.sampled_pixels)}

KEYBOARD:
u = Undo
r = Reset
c = Copy code
q = Quit
"""
            # Add quoted code and dynamic summary in stats
            code_line = self._get_mask_code_snippet(self.target_color)
            if code_line:
                wrapped = self._wrap_text(code_line, width=54)
                stats_text += "\nQuoted Mask Code (from color_shape_prep.py):\n" + wrapped + "\n"
            # Centered stats text to keep quoted line visually centered
            self.ax_stats.text(0.5, 0.97, stats_text, transform=self.ax_stats.transAxes,
                              verticalalignment='top', horizontalalignment='center',
                              fontfamily='monospace', fontsize=7)
            self.ax_stats.axis('off')
            self.ax_stats.set_title('Stats & Controls')
            
            # Store zoom state
            self.zoom_state['ax_mask'] = (mask_xlim, mask_ylim)
            
            self.fig.canvas.draw_idle()
        
        slider_h_min.on_changed(on_slider_change)
        slider_h_max.on_changed(on_slider_change)
        slider_s_min.on_changed(on_slider_change)
        slider_s_max.on_changed(on_slider_change)
        slider_v_min.on_changed(on_slider_change)
        slider_v_max.on_changed(on_slider_change)
        slider_l_min.on_changed(on_slider_change)
        slider_l_max.on_changed(on_slider_change)
        
        # Store slider references
        self.sliders = {
            'h_min': slider_h_min,
            'h_max': slider_h_max,
            's_min': slider_s_min,
            's_max': slider_s_max,
            'v_min': slider_v_min,
            'v_max': slider_v_max,
        }
        self.sliders['l_min'] = slider_l_min
        self.sliders['l_max'] = slider_l_max
    
    def update_display_partial(self):
        """Update only the sample markers without redrawing everything (preserves zoom)."""
        # Just redraw the image and mask panels while keeping their limits
        image_xlim = self.ax_image.get_xlim()
        image_ylim = self.ax_image.get_ylim()
        mask_xlim = self.ax_mask.get_xlim()
        mask_ylim = self.ax_mask.get_ylim()
        
        # Clear and redraw image
        self.ax_image.clear()
        self.im_image = self.ax_image.imshow(self.img_rgb)
        for i, (x, y) in enumerate(self.sampled_coords):
            self.ax_image.plot(x, y, 'g+', markersize=10, markeredgewidth=2)
        self.ax_image.set_title(f'Image with Samples ({len(self.sampled_pixels)})')
        self.ax_image.set_xlim(image_xlim)
        self.ax_image.set_ylim(image_ylim)
        
        # Update mask (prefer repo build when available)
        mask = None
        if build_color_masks is not None:
            try:
                hsv = self.img_hsv
                h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
                l_hls = self.img_hls[:,:,1]
                masks = build_color_masks(h, s, v, l_hls, self.img_bgr)
                key_map = {
                    'white': 'white_mask',
                    'red': 'red_mask',
                    'yellow': 'yellow_mask',
                    'blue': 'blue_mask',
                    'orange': 'orange_mask',
                    'black': 'black_mask',
                    'green': 'green_mask',
                }
                key = key_map.get(self.target_color)
                if key and key in masks:
                    mask = masks[key]
            except Exception:
                mask = None
        if mask is None:
            mask = self._build_current_mask()
        mask_display = np.zeros_like(self.img_rgb)
        mask_display[mask > 0] = [0, 255, 0]
        mask_display[mask == 0] = [200, 200, 200]
        
        self.ax_mask.clear()
        self.im_mask = self.ax_mask.imshow(mask_display)
        # Mask panel title remains simple; stats panel is updated below
        self.ax_mask.set_xlim(mask_xlim)
        self.ax_mask.set_ylim(mask_ylim)
        
        # Update HSV/HLS histograms to reflect current selection
        self.ax_hsv_hist.clear()
        self.ax_hls_hist.clear()
        if self.sampled_pixels:
            h_vals = [p[0] for p in self.sampled_pixels]
            s_vals = [p[1] for p in self.sampled_pixels]
            v_vals = [p[2] for p in self.sampled_pixels]
            self.ax_hsv_hist.bar(['H', 'S', 'V'], [np.mean(h_vals), np.mean(s_vals), np.mean(v_vals)],
                                 color=['red', 'green', 'blue'], alpha=0.7)
            self.ax_hsv_hist.set_ylim(0, 255)
            self.ax_hsv_hist.set_title(f'HSV Mean (n={len(self.sampled_pixels)})')
            self.ax_hsv_hist.text(0.02, 0.98,
                                  f"Ranges:\nH:{min(h_vals)}-{max(h_vals)} S:{min(s_vals)}-{max(s_vals)} V:{min(v_vals)}-{max(v_vals)}",
                                  transform=self.ax_hsv_hist.transAxes, va='top', fontsize=8, family='monospace')
            # HLS from coords
            hls_vals = [tuple(self.img_hls[y, x]) for (x, y) in self.sampled_coords]
            h_hls = [int(p[0]) for p in hls_vals]
            l_vals = [int(p[1]) for p in hls_vals]
            s_hls = [int(p[2]) for p in hls_vals]
            self.ax_hls_hist.bar(['H', 'L', 'S'], [np.mean(h_hls), np.mean(l_vals), np.mean(s_hls)],
                                 color=['orange', 'yellow', 'cyan'], alpha=0.7)
            self.ax_hls_hist.set_ylim(0, 255)
            self.ax_hls_hist.set_title(f'HLS Mean (n={len(self.sampled_pixels)})')
            self.ax_hls_hist.text(0.02, 0.98, f"L:{min(l_vals)}-{max(l_vals)}",
                                  transform=self.ax_hls_hist.transAxes, va='top', fontsize=8, family='monospace')
        else:
            # Keep informative defaults if no samples
            hsv = self.img_hsv
            self.ax_hsv_hist.bar(['H', 'S', 'V'], [np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])],
                                 color=['red', 'green', 'blue'], alpha=0.7)
            self.ax_hsv_hist.set_ylim(0, 255)
            self.ax_hsv_hist.set_title('HSV Mean (image)')
            hls = self.img_hls
            self.ax_hls_hist.bar(['H', 'L', 'S'], [np.mean(hls[:,:,0]), np.mean(hls[:,:,1]), np.mean(hls[:,:,2])],
                                 color=['orange', 'yellow', 'cyan'], alpha=0.7)
            self.ax_hls_hist.set_ylim(0, 255)
            self.ax_hls_hist.set_title('HLS Mean (image)')

        # Update stats panel with discrepancy metrics and selection ranges vs thresholds
        self.ax_stats.clear()
        thresh = self.thresholds
        # Count captured vs missed among sampled pixels
        captured = 0
        if len(self.sampled_coords) > 0:
            for (x, y) in self.sampled_coords:
                captured += 1 if mask[y, x] > 0 else 0
        missed = max(0, len(self.sampled_coords) - captured)

        # Selection ranges (HSV/HLS) for stats panel
        _, _, _, _, sel_h, sel_s, sel_v, sel_l = self._get_selection_ranges_and_hist()

        stats_text = (
            f"Selection HSV: H {sel_h}  S {sel_s}  V {sel_v}\n" +
            f"Selection L:   {sel_l}\n" +
            f"Samples {len(self.sampled_coords)}  Captured {captured}  Missed {missed}\n\n" +
            "Keys u=Undo r=Reset q=Quit\nMouse Scroll=Zoom Click=Sample"
        )
        # Also include quoted code
        code_line = self._get_mask_code_snippet(self.target_color)
        if code_line:
            wrapped = self._wrap_text(code_line, width=54)
            stats_text += "\nQuoted Mask Code (from color_shape_prep.py):\n" + wrapped + "\n"
        # Centered stats text to keep quoted line visually centered
        self.ax_stats.text(0.5, 0.97, stats_text, transform=self.ax_stats.transAxes,
                           va='top', ha='center', fontfamily='monospace', fontsize=7)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Stats & Controls')
        
        # Store zoom state
        self.zoom_state['ax_image'] = (image_xlim, image_ylim)
        self.zoom_state['ax_mask'] = (mask_xlim, mask_ylim)
        
        self.fig.canvas.draw_idle()
    
    def _build_current_mask(self):
        """Build mask using build_color_masks directly for the selected color."""
        h, s, v = cv2.split(self.img_hsv)
        hls_channels = cv2.split(self.img_hls)
        l_hls = hls_channels[1]
        img = self.img_bgr
        masks = build_color_masks(h, s, v, l_hls, img)
        key_map = {
            'white': 'white_mask',
            'red': 'red_mask',
            'yellow': 'yellow_mask',
            'blue': 'blue_mask',
            'orange': 'orange_mask',
            'black': 'black_mask',
            'green': 'green_mask',
        }
        mask = masks.get(key_map.get(self.target_color))
        if mask is not None:
            return mask.astype(np.uint8) * 255
        # Fallback to zeros if all else fails
        return np.zeros(h.shape, dtype=np.uint8)

    def _get_mask_code_snippet(self, color: str) -> str:
        """Return the quoted mask assignment line(s) from color_shape_prep.py for the given color."""
        try:
            from pathlib import Path
            path = Path('src') / 'color_shape_prep.py'
            if not path.exists():
                return ''
            with open(path, 'r') as f:
                lines = f.readlines()
            # Find build_color_masks and collect its block
            start = None
            for i, line in enumerate(lines):
                if line.strip().startswith('def build_color_masks('):
                    start = i
                    break
            if start is None:
                return ''
            block = lines[start:start+300]
            key_map = {
                'red': 'red_mask',
                'yellow': 'yellow_mask',
                'blue': 'blue_mask',
                'orange': 'orange_mask',
                'white': 'white_mask',
                'black': 'black_mask',
                'green': 'green_mask',
            }
            target = key_map.get(color)
            if not target:
                return ''
            # Extract the target assignment; if multi-line, collapse to a single line
            i = 0
            while i < len(block):
                t = block[i].strip()
                if t.startswith(target + ' ='):
                    # Accumulate subsequent lines if the assignment spans multiple lines
                    assignment_lines = [t]
                    j = i + 1
                    # Continue until we likely hit end of expression (heuristic: balanced parentheses or next assignment)
                    paren_balance = t.count('(') - t.count(')')
                    while j < len(block) and paren_balance > 0:
                        line_j = block[j].strip()
                        assignment_lines.append(line_j)
                        paren_balance += line_j.count('(') - line_j.count(')')
                        j += 1
                    # Collapse to single line
                    single = ' '.join(assignment_lines)
                    # Remove redundant spaces
                    single = ' '.join(single.split())
                    return single
                i += 1
            return ''
        except Exception:
            return ''

    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap a single long line of text at spaces to the given width.

        Keeps operators and parentheses readable; avoids breaking mid-token.
        """
        if not text:
            return ''
        import textwrap
        # Ensure it's a single line and collapse excessive spaces
        single = ' '.join(text.replace('\n', ' ').split())
        # Wrap with break_long_words=False to avoid splitting tokens
        wrapped_lines = textwrap.wrap(single, width=width, break_long_words=False, break_on_hyphens=False)
        return '\n'.join(wrapped_lines)

    
    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive color mask adjuster')
    parser.add_argument('image_name', help='Image filename or basename; script will attempt to locate it')
    parser.add_argument('--color', default='white', 
                       choices=['white', 'red', 'yellow', 'blue', 'orange', 'black', 'green'],
                       help='Color to adjust')
    parser.add_argument('--skip-wb', action='store_true', help='Skip gray-world white balance (use raw image)')

    args = parser.parse_args()

    def find_image_path(name: str) -> str:
        """Resolve image path similar to color-shape tools: direct or kaggle images.

        - If `name` is an existing file path, return it.
        - Else, look for basename match under `data/raw/kaggle_roadsign/images`.
        """
        p = Path(name)
        if p.is_file():
            return str(p)

        images_dir = Path('data') / 'raw' / 'kaggle_roadsign' / 'images'
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Build candidate names: exact name, plus common extensions if none provided
        candidates = [name]
        if p.suffix == '':
            for ext in ('.png', '.jpg', '.jpeg'):
                candidates.append(name + ext)

        # Search only within the images directory
        for f in images_dir.rglob('*'):
            if not f.is_file():
                continue
            if f.name in candidates or f.stem == p.stem:
                print(f"Found image: {f} (using first match)")
                return str(f)

        raise FileNotFoundError(f"Image '{name}' not found in {images_dir}")

    image_path = find_image_path(args.image_name)
    adjuster = ColorMaskAdjuster(image_path, target_color=args.color, skip_wb=args.skip_wb)
    adjuster.show()


if __name__ == '__main__':
    main()
