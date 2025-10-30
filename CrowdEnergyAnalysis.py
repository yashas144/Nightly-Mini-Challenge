"""
Travis Scott - FEIN Concert Energy Analyzer
Complete implementation for the challenge

Setup:
1. Install dependencies:
   pip install opencv-python numpy matplotlib scipy yt-dlp

2. Run:
   python travis_scott_analyzer.py
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
import os
import subprocess
import sys

class TravisScottEnergyAnalyzer:
    def __init__(self, video_path=None):
        if video_path is None:
            print(" Downloading Travis Scott - FEIN from YouTube...")
            video_path = self.download_video()
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\n Video loaded successfully!")
        print(f" Duration: {self.duration:.1f}s @ {self.fps:.1f} fps")
        print(f" Total frames: {self.total_frames}")
    
    def download_video(self):
        """Download the Travis Scott concert video using yt-dlp."""
        try:
            output_file = "travis_scott_fein.mp4"
            
            if os.path.exists(output_file):
                print(f" Video already downloaded: {output_file}")
                return output_file
            
            # Download first 60 seconds
            cmd = [
                'yt-dlp',
                'https://www.youtube.com/watch?v=L3J5DKwiVms',
                '-f', 'best[height<=720]',  # 720p max for faster processing
                '-o', output_file,
                '--download-sections', '*0-60',  # First 60 seconds only
                '--force-keyframes-at-cuts'
            ]
            
            print("Downloading video (this may take a minute)...")
            subprocess.run(cmd, check=True)
            print(f" Downloaded: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError:
            print("\n Error: yt-dlp download failed.")
            print("Please install yt-dlp: pip install yt-dlp")
            print("Or download the video manually and pass the path as argument.")
            sys.exit(1)
        except FileNotFoundError:
            print("\n Error: yt-dlp not found.")
            print("Install it with: pip install yt-dlp")
            sys.exit(1)
    
    def analyze_crowd_energy(self, sample_rate=2):
        """
        Analyze crowd energy using optical flow and visual metrics.
        sample_rate=2 for high-energy concerts (Travis Scott needs high sensitivity!)
        """
        print("\n Analyzing FEIN concert energy...")
        print("This will take 2-3 minutes...")
        
        energy_scores = []
        frame_times = []
        prev_gray = None
        frame_count = 0
        
        # Progress tracking
        progress_interval = self.total_frames // 20
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
            
            # Resize for faster processing
            frame_small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Optical flow - detects crowd movement
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Calculate motion magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_score = np.mean(magnitude)
                
                # Visual complexity - lights, pyro, crowd activity
                color_var = np.std(frame_small)
                
                # Edge detection - high for jumping crowds
                edges = cv2.Canny(gray, 50, 150)
                edge_score = np.mean(edges) / 255.0
                
                # Combined energy score (optimized for concerts)
                energy = (
                    motion_score * 0.6 +      # Movement is key
                    color_var * 0.002 +       # Visual activity
                    edge_score * 0.4          # Crowd definition
                )
                
                energy_scores.append(energy)
                frame_times.append(frame_count / self.fps)
            
            prev_gray = gray
            frame_count += 1
            
            # Progress update
            if frame_count % progress_interval == 0:
                progress = (frame_count / self.total_frames) * 100
                print(f"   Progress: {progress:.0f}% ({frame_count}/{self.total_frames} frames)")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset for later use
        
        # Smooth the energy curve
        if len(energy_scores) > 10:
            energy_scores = gaussian_filter1d(energy_scores, sigma=1.5)
        
        print(" Energy analysis complete!")
        return np.array(frame_times), np.array(energy_scores)
    
    def find_low_energy_moments(self, times, energy, threshold_percentile=35):
        """Find dips in energy for show cue suggestions."""
        threshold = np.percentile(energy, threshold_percentile)
        
        low_moments = []
        in_low_zone = False
        low_start = 0
        
        for i, (t, e) in enumerate(zip(times, energy)):
            if e < threshold and not in_low_zone:
                in_low_zone = True
                low_start = i
            elif e >= threshold and in_low_zone:
                in_low_zone = False
                duration = times[i] - times[low_start]
                
                # Only consider sustained dips (>1.5 seconds)
                if duration > 1.5:
                    mid_idx = (low_start + i) // 2
                    low_moments.append({
                        'time': times[mid_idx],
                        'energy': energy[mid_idx],
                        'duration': duration
                    })
        
        return low_moments
    
    def generate_travis_scott_cues(self, low_moments):
        """Generate show cues specifically for Travis Scott's style."""
        # Travis Scott show aesthetic: dark, explosive, pyro-heavy
        cue_types = [
            " Lighting: Blackout → Red strobe explosion on beat drop",
            " Pyro: Flame columns with CO2 burst, build crowd tension",
            " Visual: Cactus logo projection + laser grid sweep",
            " Lighting: Blue/purple wash, slow intensity crescendo",
            " Visual: LED wall glitch effect, prepare for rage moment",
            " Pyro: Dual flame throwers + strobe, maximum intensity",
            " Visual: Confetti + smoke machine, crowd interaction peak",
            " Lighting: Pulsing red with blackout intervals, heartbeat effect",
            " Visual: Spotlight on crowd sections, encourage mosh pit",
            " Pyro: Firework sequence + full stage strobe reset"
        ]
        
        cues = []
        for i, moment in enumerate(low_moments):
            cue = {
                'timestamp': f"{moment['time']:.1f}s",
                'energy_level': f"{moment['energy']:.2f}",
                'suggestion': cue_types[i % len(cue_types)],
                'reasoning': f"Energy dip detected - {moment['duration']:.1f}s duration",
                'context': 'Build tension before next drop'
            }
            cues.append(cue)
        
        return cues
    
    def find_peak_moment(self, times, energy, window_seconds=10):
        """Find the most intense 10-second window (likely during FEIN drop)."""
        if len(times) < 2:
            return None
            
        window_frames = int(window_seconds / (times[1] - times[0]))
        
        if window_frames >= len(energy):
            window_frames = len(energy) - 1
        
        best_score = 0
        best_start_idx = 0
        
        for i in range(len(energy) - window_frames):
            window_energy = np.mean(energy[i:i+window_frames])
            if window_energy > best_score:
                best_score = window_energy
                best_start_idx = i
        
        start_time = times[best_start_idx]
        end_time = min(start_time + window_seconds, self.duration)
        
        return {
            'start': start_time,
            'end': end_time,
            'avg_energy': best_score,
            'description': 'Peak crowd energy moment (likely FEIN drop/mosh pit)'
        }
    
    def extract_highlight_clip(self, start_time, end_time, output_path="highlight_fein.mp4"):
        """Extract the 10-second highlight clip."""
        print(f"\n Extracting highlight clip...")
        print(f"  {start_time:.1f}s → {end_time:.1f}s")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * self.fps))
        
        # Get frame dimensions
        ret, frame = self.cap.read()
        if not ret:
            print(" Could not read frame for highlight")
            return
        
        height, width = frame.shape[:2]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Reset to start time
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * self.fps))
        
        current_time = start_time
        frame_count = 0
        
        while current_time < end_time:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Add highlight label
            label_text = "HIGHLIGHT MOMENT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (40, 30),
                (40 + text_width + 20, 30 + text_height + 20),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, label_text,
                (50, 50 + text_height),
                font, font_scale,
                (0, 255, 255),
                thickness
            )
            
            out.write(frame)
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_count += 1
        
        out.release()
        print(f" Saved highlight clip: {output_path}")
        print(f" Clip duration: {frame_count / self.fps:.1f}s")
    
    def visualize_energy(self, times, energy, low_moments, peak_moment):
        """Create the energy visualization graph."""
        try:
            import matplotlib.pyplot as plt
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(16, 7))
            
            # Plot energy curve
            ax.plot(times, energy, linewidth=3, 
                   color='#8b5cf6', label='Crowd Energy',
                   marker='o', markersize=3, markevery=5)
            
            # Fill under curve
            ax.fill_between(times, energy, alpha=0.3, color='#8b5cf6')
            
            # Mark low energy moments (cue points)
            for moment in low_moments:
                ax.axvline(moment['time'], color='#f59e0b',
                          linestyle='--', alpha=0.7, linewidth=2)
                ax.text(moment['time'], ax.get_ylim()[1] * 0.95,
                       ' ', fontsize=16, ha='center')
            
            # Highlight peak moment
            if peak_moment:
                ax.axvspan(peak_moment['start'], peak_moment['end'],
                          alpha=0.3, color='#10b981', label='Peak Moment')
                ax.text((peak_moment['start'] + peak_moment['end']) / 2,
                       ax.get_ylim()[1] * 0.85,
                       ' PEAK\nHIGHLIGHT', fontsize=14,
                       ha='center', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='#10b981', alpha=0.8))
            
            # Styling
            ax.set_xlabel('Time (seconds)', fontsize=14, weight='bold')
            ax.set_ylabel('Energy Level', fontsize=14, weight='bold')
            ax.set_title('Travis Scott - FEIN | Crowd Energy Analysis',
                        fontsize=18, weight='bold', pad=20)
            ax.legend(fontsize=12, loc='upper left')
            ax.grid(True, alpha=0.2, linestyle='--')
            
            plt.tight_layout()
            plt.savefig('energy_analysis_fein.png', dpi=150,
                       bbox_inches='tight', facecolor='#1a1a1a')
            print(" Energy visualization saved: energy_analysis_fein.png")
            plt.close()
            
        except ImportError:
            print("  Matplotlib not installed, skipping visualization")
    
    def run_complete_analysis(self):
        """Run the full analysis pipeline."""
        print("\n" + "="*60)
        print(" TRAVIS SCOTT - FEIN | ENERGY ANALYSIS")
        print(" CIRCUS MAXIMUS MILANO")
        print("="*60)
        
        # Step 1: Analyze energy
        times, energy = self.analyze_crowd_energy(sample_rate=2)
        
        # Step 2: Find low energy moments
        low_moments = self.find_low_energy_moments(times, energy, threshold_percentile=35)
        print(f"\n Found {len(low_moments)} low-energy moments for show cues")
        
        # Step 3: Generate show cues
        cues = self.generate_travis_scott_cues(low_moments)
        
        # Step 4: Find peak moment
        peak = self.find_peak_moment(times, energy, window_seconds=10)
        
        # Step 5: Create visualization
        self.visualize_energy(times, energy, low_moments, peak)
        
        # Step 6: Save results
        results = {
            'video': 'Travis Scott - FEIN LIVE @ CIRCUS MAXIMUS MILANO',
            'url': 'https://www.youtube.com/watch?v=L3J5DKwiVms',
            'analysis': {
                'duration': f"{self.duration:.1f}s",
                'fps': self.fps,
                'total_data_points': len(times)
            },
            'show_cues': cues,
            'peak_moment': {
                'start': f"{peak['start']:.1f}s",
                'end': f"{peak['end']:.1f}s",
                'avg_energy': f"{peak['avg_energy']:.2f}",
                'description': peak['description']
            }
        }
        
        with open('fein_analysis_results.json', 'w') as f:
            json.dump(results, indent=2, fp=f)
        print(" Results saved: fein_analysis_results.json")
        
        # Step 7: Print summary
        print("\n" + "="*60)
        print(" SHOW CUE SUGGESTIONS")
        print("="*60)
        for i, cue in enumerate(cues[:5], 1):
            print(f"\n{i}.   @ {cue['timestamp']}")
            print(f"   {cue['suggestion']}")
            print(f"   └─ {cue['reasoning']}")
            print(f"   └─ Energy: {cue['energy_level']}")
        
        if len(cues) > 5:
            print(f"\n   ... and {len(cues)-5} more cues (see JSON)")
        
        print("\n" + "="*60)
        print(" PEAK MOMENT - HIGHLIGHT CLIP")
        print("="*60)
        print(f" Timeframe: {peak['start']:.1f}s → {peak['end']:.1f}s")
        print(f" Average Energy: {peak['avg_energy']:.2f}")
        print(f" {peak['description']}")
        
        # Step 8: Extract highlight
        self.extract_highlight_clip(peak['start'], peak['end'])
        
        self.cap.release()
        
        print("\n" + "="*60)
        print(" ANALYSIS COMPLETE!")
        print("="*60)
        print("\n Output Files:")
        print("   1. fein_analysis_results.json  - Complete analysis data")
        print("   2. energy_analysis_fein.png    - Energy graph visualization")
        print("   3. highlight_fein.mp4          - 10-second peak moment clip")
        print("\n Ready for your Loom demo recording!")
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze Travis Scott FEIN concert energy'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file (downloads from YouTube if not provided)'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = TravisScottEnergyAnalyzer(args.video)
        results = analyzer.run_complete_analysis()
        
    except KeyboardInterrupt:
        print("\n\n Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
























