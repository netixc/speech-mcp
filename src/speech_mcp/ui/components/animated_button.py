"""
Animated button component for the Speech UI.

This module provides a PyQt button with press animation effects.
"""

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QTimer

class AnimatedButton(QPushButton):
    """
    Custom button class with press animation effect.
    """
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFlat(False)
        self.original_style = ""
        self.animation_timer = None
        self.animation_step = 0
        self.animation_steps = 10
        self.animation_direction = 1  # 1 for pressing, -1 for releasing
        self.is_animating = False
        
    def set_style(self, style):
        """Set the button's base style"""
        self.original_style = style
        self.setStyleSheet(style)
        
    def mousePressEvent(self, event):
        """Handle mouse press event with animation"""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            
        self.animation_step = 0
        self.animation_direction = 1
        self.is_animating = True
        
        # Start animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_press)
        self.animation_timer.start(20)  # 20ms per frame
        
        # Call parent handler
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release event with animation"""
        if self.animation_timer is not None:
            self.animation_timer.stop()
            
        self.animation_step = self.animation_steps
        self.animation_direction = -1
        self.is_animating = True
        
        # Start animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate_release)
        self.animation_timer.start(20)  # 20ms per frame
        
        # Call parent handler
        super().mouseReleaseEvent(event)
        
    def animate_press(self):
        """Animate button press"""
        if not self.is_animating:
            return
            
        self.animation_step += 1
        
        if self.animation_step >= self.animation_steps:
            self.animation_step = self.animation_steps
            self.animation_timer.stop()
            
        # Calculate animation progress (0.0 to 1.0)
        progress = self.animation_step / self.animation_steps
        
        # Apply visual changes based on progress
        self._apply_animation_style(progress)
        
    def animate_release(self):
        """Animate button release"""
        if not self.is_animating:
            return
            
        self.animation_step -= 1
        
        if self.animation_step <= 0:
            self.animation_step = 0
            self.animation_timer.stop()
            self.is_animating = False
            self.setStyleSheet(self.original_style)
            return
            
        # Calculate animation progress (1.0 to 0.0)
        progress = self.animation_step / self.animation_steps
        
        # Apply visual changes based on progress
        self._apply_animation_style(progress)
        
    def _apply_animation_style(self, progress):
        """Apply animation style based on progress (0.0 to 1.0)"""
        if not self.original_style:
            return
            
        # Extract background color from original style
        bg_color = None
        for style_part in self.original_style.split(";"):
            if "background-color:" in style_part:
                bg_color = style_part.split(":")[1].strip()
                break
                
        if not bg_color:
            return
            
        # Parse color
        if bg_color.startswith("#"):
            # Hex color
            r = int(bg_color[1:3], 16)
            g = int(bg_color[3:5], 16)
            b = int(bg_color[5:7], 16)
            
            # Darken for press effect (reduce by up to 30%)
            darken_factor = 0.7 + (0.3 * (1.0 - progress))
            r = max(0, int(r * darken_factor))
            g = max(0, int(g * darken_factor))
            b = max(0, int(b * darken_factor))
            
            # Create new style with darkened color
            new_bg_color = f"#{r:02x}{g:02x}{b:02x}"
            new_style = self.original_style.replace(bg_color, new_bg_color)
            
            # Add slight inset shadow effect
            shadow_strength = int(progress * 5)
            if shadow_strength > 0:
                new_style += f"; border: none; border-radius: 5px; padding: 8px 16px; font-weight: bold; margin: {shadow_strength}px 0px 0px 0px;"
                
            self.setStyleSheet(new_style)