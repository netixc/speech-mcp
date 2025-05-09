"""
Audio visualizer component for the Speech UI.

This module provides a PyQt widget for visualizing audio levels and waveforms.
Optimized for lower CPU usage and better performance.
"""

import time
import math
import random
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QLinearGradient, QBrush

class AudioVisualizer(QWidget):
    """
    Widget for visualizing audio levels and waveforms.
    Optimized for lower resource usage.
    """
    def __init__(self, parent=None, mode="user", width_factor=1.0):
        """
        Initialize the audio visualizer.
        
        Args:
            parent: Parent widget
            mode: "user" or "agent" to determine color scheme
            width_factor: Factor to adjust the width of bars (1.0 = full width, 0.5 = half width)
        """
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.audio_levels = [0.0] * 20  # Reduced buffer size (was 50)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.mode = mode
        self.width_factor = width_factor
        self.active = False  # Track if visualizer is active
        
        # Set colors based on mode
        if self.mode == "user":
            self.bar_color = QColor(0, 200, 255, 180)  # Blue for user
            self.glow_color = QColor(0, 120, 255, 80)  # Softer blue glow
            # Less smoothing for user mode to be more responsive
            self.smoothing_factor = 0.15
        else:
            self.bar_color = QColor(0, 255, 100, 200)  # Brighter green for agent
            self.glow_color = QColor(0, 220, 100, 100)  # Stronger green glow
            # More smoothing for agent mode
            self.smoothing_factor = 0.3
            
        # Inactive colors (grey)
        self.inactive_bar_color = QColor(100, 100, 100, 120)  # Grey for inactive
        self.inactive_glow_color = QColor(80, 80, 80, 60)  # Softer grey glow
        
        # Add a smoothing factor to make the visualization less jumpy
        self.last_level = 0.0
        
        # Animation time for dynamic effects
        self.animation_time = 0.0
        
        # Frame rate control
        if self.mode == "user":
            self.fps = 30  # Higher frame rate for user visualizer (more responsive)
        else:
            self.fps = 24  # Standard frame rate for agent visualizer
            
        self.last_update_time = time.time()
        
        # Timer for animation - single timer approach
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(1000 // self.fps)  # ms between frames
        
        # Pre-recorded animation patterns for agent mode (simplified)
        if self.mode == "agent":
            self.initialize_prerecorded_patterns()
            self.current_pattern = "wave"  # Default pattern
    
    def update_animation(self):
        """Update the animation state and trigger a repaint if needed"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Only update if enough time has passed (frame limiting)
        if elapsed >= (1.0 / self.fps):
            self.last_update_time = current_time
            
            # Only request a repaint if the visualizer is active or in agent mode
            if self.active or (self.mode == "agent" and hasattr(self, 'patterns')):
                self.update()  # Request a repaint
    
    def initialize_prerecorded_patterns(self):
        """Initialize pre-recorded animation patterns for agent visualization"""
        # Create different animation patterns (simplified)
        self.patterns = {
            "wave": self.generate_wave_pattern(),
            "pulse": self.generate_pulse_pattern()
        }
    
    def generate_wave_pattern(self):
        """Generate a smooth wave pattern (optimized)"""
        pattern = []
        # Create a smooth sine wave pattern (reduced steps)
        steps = 20  # Reduced from 40
        for i in range(steps):
            # Sine wave with varying amplitude
            angle = (i / steps) * (2 * math.pi)
            level = 0.3 + 0.4 * math.sin(angle)
            pattern.append(level)
        return pattern
    
    def generate_pulse_pattern(self):
        """Generate a pulsing pattern (optimized)"""
        pattern = []
        # Create a pulsing pattern (reduced steps)
        steps = 15  # Reduced from 30
        for i in range(steps):
            # Pulse wave (higher in middle)
            position = i / steps
            if position < 0.5:
                level = 0.3 + 1.2 * position  # Rising
            else:
                level = 0.3 + 1.2 * (1.0 - position)  # Falling
            pattern.append(level * 0.7)  # Scale to appropriate range
        return pattern
    
    def set_active(self, active):
        """Set the visualizer as active or inactive."""
        self.active = active
        if active and self.mode == "agent":
            # Randomly select a pattern
            patterns = list(self.patterns.keys())
            self.current_pattern = random.choice(patterns)
        elif not active:
            # Reset audio levels when deactivating
            self.audio_levels = [0.0] * len(self.audio_levels)
            self.last_level = 0.0
            
        self.update()  # Request a single repaint
    
    def update_level(self, level):
        """Update with a new audio level."""
        if self.mode == "agent" and hasattr(self, 'patterns'):
            # For agent mode, we use pre-recorded patterns instead of audio input
            return
            
        # For user mode, we still use the audio input
        # Apply smoothing to avoid abrupt changes
        
        # Amplify the user audio level (increase by 70%)
        if self.mode == "user":
            level = min(1.0, level * 1.7)  # Amplify but cap at 1.0
            
        smoothed_level = (level * (1.0 - self.smoothing_factor)) + (self.last_level * self.smoothing_factor)
        self.last_level = smoothed_level
        
        # Update the audio levels buffer
        self.audio_levels.pop(0)
        self.audio_levels.append(smoothed_level)
        
        # We don't force an update here - let the timer handle it
    
    def paintEvent(self, event):
        """Draw the audio visualization (optimized)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)  # Enable only when needed
        
        # Draw background
        painter.fillRect(event.rect(), QColor(30, 30, 30))
        
        # Get dimensions
        width = self.width()
        height = self.height()
        mid_height = height / 2
        
        # Choose colors based on active state
        if self.active:
            bar_color = self.bar_color
            glow_color = self.glow_color
        else:
            bar_color = self.inactive_bar_color
            glow_color = self.inactive_glow_color
        
        # Optimize: Reduced bar count
        bar_count = 20  # Reduced from 40
        bar_width = (width / bar_count) * self.width_factor
        bar_spacing = 2  # Pixels between bars
        
        # Calculate animation phase (simplified)
        phase = time.time() % (2 * math.pi)
        
        # Reusable colors for performance
        bar_colors = []
        glow_colors = []
        for i in range(bar_count // 2):
            bar_colors.append(QColor(
                bar_color.red(), 
                bar_color.green(), 
                bar_color.blue(), 
                180 - i * 6  # Adjusted for fewer bars
            ))
            glow_colors.append(QColor(
                glow_color.red(), 
                glow_color.green(), 
                glow_color.blue(), 
                80 - i * 4  # Adjusted for fewer bars
            ))
        
        # Special handling for agent mode with pre-recorded patterns
        if self.mode == "agent" and hasattr(self, 'patterns') and self.active:
            pattern = self.patterns[self.current_pattern]
            pattern_length = len(pattern)
            
            # Use time-based animation
            time_index = int((time.time() * 8) % pattern_length)  # Reduced speed from 10 to 8
            
            # Pre-calculate levels for efficiency
            levels = []
            for i in range(bar_count // 2):
                # Calculate pattern index with wrapping
                pattern_idx = (time_index + i) % pattern_length
                base_level = pattern[pattern_idx]
                
                # Add subtle wave effect (simplified)
                wave_effect = 0.05 * math.sin(phase + i * 0.2)
                level = max(0.0, min(1.0, base_level + wave_effect))
                levels.append(level)
            
            # Draw bars (batch processing)
            for i in range(bar_count // 2):
                level = levels[i]
                bar_height = level * mid_height * 0.95
                
                # Calculate positions
                x = (width / 2) + (i * bar_width / 2) - (bar_width / 2)
                x_mirror = (width / 2) - (i * bar_width / 2) - (bar_width / 2)
                
                # Draw right side (main bar)
                rect = QRectF(x, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect, bar_colors[i])
                
                # Draw left side (main bar)
                rect_mirror = QRectF(x_mirror, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect_mirror, bar_colors[i])
                
                # Only draw glow for the first few bars (optimization)
                if i < 5:
                    # Draw right side glow
                    glow_rect = QRectF(
                        x - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect, glow_colors[i])
                    
                    # Draw left side glow
                    glow_rect_mirror = QRectF(
                        x_mirror - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect_mirror, glow_colors[i])
        else:
            # User mode or inactive agent mode (optimized)
            for i in range(bar_count // 2):
                # Get level from audio buffer
                level_idx = len(self.audio_levels) - 1 - i
                if 0 <= level_idx < len(self.audio_levels):
                    level = self.audio_levels[level_idx]
                else:
                    level = 0.0
                    
                # If inactive, flatten the visualization
                if not self.active:
                    level = level * 0.2
                
                # Add subtle wave effect (simplified)
                if i < 5:  # Only add effect to first few bars
                    wave_effect = 0.05 * math.sin(phase + i * 0.2)
                    level = max(0.0, min(1.0, level + wave_effect))
                
                # Calculate bar height
                bar_height = level * mid_height * 0.95
                
                # Calculate positions
                x = (width / 2) + (i * bar_width / 2) - (bar_width / 2)
                x_mirror = (width / 2) - (i * bar_width / 2) - (bar_width / 2)
                
                # Draw right side (main bar)
                rect = QRectF(x, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect, bar_colors[i] if i < len(bar_colors) else bar_color)
                
                # Draw left side (main bar)
                rect_mirror = QRectF(x_mirror, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                painter.fillRect(rect_mirror, bar_colors[i] if i < len(bar_colors) else bar_color)
                
                # Only draw glow for the first few bars (optimization)
                if i < 5:
                    # Draw right side glow
                    glow_rect = QRectF(
                        x - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect, glow_colors[i] if i < len(glow_colors) else glow_color)
                    
                    # Draw left side glow
                    glow_rect_mirror = QRectF(
                        x_mirror - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect_mirror, glow_colors[i] if i < len(glow_colors) else glow_color)
        
        # Draw a thin center line with a gradient (simplified)
        if self.active:  # Only draw when active
            gradient = QLinearGradient(0, mid_height, width, mid_height)
            gradient.setColorAt(0, QColor(100, 100, 100, 0))
            gradient.setColorAt(0.5, QColor(100, 100, 100, 100))
            gradient.setColorAt(1, QColor(100, 100, 100, 0))
            
            painter.setPen(QPen(gradient, 1))
            painter.drawLine(0, int(mid_height), width, int(mid_height))