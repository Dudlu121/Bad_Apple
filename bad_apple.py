import cv2
import os
import sys
import numpy as np
import time
from moviepy.editor import VideoFileClip, AudioFileClip

FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# --- Helper Function to find videos in the current directory ---
def get_video_files():
    """Returns a list of all video files in the current directory."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in os.listdir('.') if f.endswith(video_extensions)]

def on_trackbar(val):
    """Callback function for the trackbar. Does nothing, but is required."""
    pass

def adjust_thresholds(bad_apple_video_path, fg_video_path, bg_video_path, start_second, duration):
    """
    Adjusts the foreground and background thresholds for the video processing.
    """
    fg_cap = cv2.VideoCapture(fg_video_path)
    bg_cap = cv2.VideoCapture(bg_video_path)
    bad_apple_cap = cv2.VideoCapture(bad_apple_video_path)

    if not fg_cap.isOpened() or not bg_cap.isOpened() or not bad_apple_cap.isOpened():
        print("Error: Could not open one or more video files for threshold adjustment.")
        return 127, 127
    
    fps = bad_apple_cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_second)
    end_frame = int(fps * (start_second + duration))
    
    bad_apple_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fg_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fg_total_frames = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_total_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad_apple_total_frames = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('Adjust Thresholds')
    cv2.createTrackbar('FG Threshold', 'Adjust Thresholds', 127, 255, on_trackbar)
    cv2.createTrackbar('BG Threshold', 'Adjust Thresholds', 127, 255, on_trackbar)
    
    print(f"\nAdjust the trackbars to find the best thresholds for your videos ({start_second}-{start_second+duration}s).")
    print("Press 'q' to finalize and close.")

    frame_number_offset = start_frame
    while True:
        ret_apple, apple_frame = bad_apple_cap.read()
        
        if not ret_apple:
            # Loop the video if it ends within the preview duration
            bad_apple_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret_apple, apple_frame = bad_apple_cap.read()
            
        fg_threshold = cv2.getTrackbarPos('FG Threshold', 'Adjust Thresholds')
        bg_threshold = cv2.getTrackbarPos('BG Threshold', 'Adjust Thresholds')
        
        gray_apple = cv2.cvtColor(apple_frame, cv2.COLOR_BGR2GRAY)
        
        _, fg_mask = cv2.threshold(gray_apple, fg_threshold, 255, cv2.THRESH_BINARY)
        _, bg_mask = cv2.threshold(gray_apple, bg_threshold, 255, cv2.THRESH_BINARY_INV)

        fg_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_offset)
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_offset)
        
        ret_fg, fg_frame = fg_cap.read()
        ret_bg, bg_frame = bg_cap.read()
        
        if not ret_fg or not ret_bg:
            fg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_fg, fg_frame = fg_cap.read()
            ret_bg, bg_frame = bg_cap.read()
        
        fg_frame_resized = cv2.resize(fg_frame, (width, height))
        bg_frame_resized = cv2.resize(bg_frame, (width, height))
        
        fg_masked = cv2.bitwise_and(fg_frame_resized, fg_frame_resized, mask=fg_mask)
        bg_masked = cv2.bitwise_and(bg_frame_resized, bg_frame_resized, mask=bg_mask)
        
        final_frame = cv2.add(fg_masked, bg_masked)
        cv2.imshow('Adjust Thresholds', final_frame)

        frame_number_offset += 1
        if frame_number_offset >= end_frame:
            frame_number_offset = start_frame
            bad_apple_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    fg_cap.release()
    bg_cap.release()
    bad_apple_cap.release()
    
    return fg_threshold, bg_threshold

def preview_bad_apple_effect(bad_apple_video_path, fg_video_path, bg_video_path, seconds, fg_threshold, bg_threshold):
    """Generates and plays a preview of the final blended video by first rendering to a temporary file."""
    bad_apple_cap = cv2.VideoCapture(bad_apple_video_path)
    fg_cap = cv2.VideoCapture(fg_video_path)
    bg_cap = cv2.VideoCapture(bg_video_path)

    if not bad_apple_cap.isOpened() or not fg_cap.isOpened() or not bg_cap.isOpened():
        print("Error: Could not open one or more video files for preview.")
        return False

    width = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = bad_apple_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * seconds)

    fg_total_frames = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_total_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad_apple_total_frames = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_preview_path = "temp_preview.mp4"
    preview_video_writer = cv2.VideoWriter(temp_preview_path, FOURCC, fps, (width, height))
    
    print("\nRendering a 10-second preview... please wait.")
    start_time = time.time() # Start the timer for the preview rendering

    for i in range(total_frames):
        ret_apple, apple_frame = bad_apple_cap.read()
        
        if not ret_apple:
            break

        fg_frame_to_read = int(i * (fg_total_frames / bad_apple_total_frames))
        bg_frame_to_read = int(i * (bg_total_frames / bad_apple_total_frames))

        fg_cap.set(cv2.CAP_PROP_POS_FRAMES, fg_frame_to_read)
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_to_read)
        
        ret_fg, fg_frame = fg_cap.read()
        ret_bg, bg_frame = bg_cap.read()

        if not ret_fg or not ret_bg:
            fg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_fg, fg_frame = fg_cap.read()
            ret_bg, bg_frame = bg_cap.read()

        fg_frame_resized = cv2.resize(fg_frame, (width, height))
        bg_frame_resized = cv2.resize(bg_frame, (width, height))
        
        gray_apple = cv2.cvtColor(apple_frame, cv2.COLOR_BGR2GRAY)
        
        _, fg_mask = cv2.threshold(gray_apple, fg_threshold, 255, cv2.THRESH_BINARY)
        _, bg_mask = cv2.threshold(gray_apple, bg_threshold, 255, cv2.THRESH_BINARY_INV)

        fg_masked = cv2.bitwise_and(fg_frame_resized, fg_frame_resized, mask=fg_mask)
        bg_masked = cv2.bitwise_and(bg_frame_resized, bg_frame_resized, mask=bg_mask)
        
        final_frame = cv2.add(fg_masked, bg_masked)
        
        preview_video_writer.write(final_frame)
        
        # Print progress and ETA
        if (i + 1) % 10 == 0 or i + 1 == total_frames:
            elapsed_time = time.time() - start_time
            progress = (i + 1) / total_frames
            
            if progress > 0:
                eta_seconds = (elapsed_time / progress) - elapsed_time
                minutes = int(eta_seconds // 60)
                seconds = int(eta_seconds % 60)
                sys.stdout.write(f"\rProgress: {progress*100:.2f}% | ETA: {minutes}m {seconds}s")
                sys.stdout.flush()

    bad_apple_cap.release()
    fg_cap.release()
    bg_cap.release()
    preview_video_writer.release()
    
    print("\nPreview rendered. Now playing...")
    
    # Play back the temporary preview video
    preview_clip = VideoFileClip(temp_preview_path)
    preview_clip.preview()
    preview_clip.close()
    
    os.remove(temp_preview_path)

    return True

# --- Main Application Logic ---
def main():
    """
    The main function of the Bad Apple application.
    It guides the user through selecting videos, processes the frames, and
    creates the final output video.
    """
    print("IF A VIDEO EXIST IT CAN BE BAD APPLE !\n")
    print("Internet.")
    
    # Get a list of all available videos in the directory
    available_videos = get_video_files()

    if len(available_videos) < 3:
        print("Error: Please make sure you have at least three video files in this directory:")
        print("  1. The Bad Apple mask video")
        print("  2. A foreground video")
        print("  3. A background video")
        sys.exit(1)
    
    # Display the list of videos for the user
    print("\nAvailable videos in this directory:")
    for i, file_name in enumerate(available_videos):
        print(f"  [{i+1}] {file_name}")

    # Prompt user for video selections with error handling
    try:
        # Step 1: Select the Bad Apple mask video
        bad_apple_choice_idx = int(input("\nEnter the number for the BAD APPLE video: ")) - 1
        if not (0 <= bad_apple_choice_idx < len(available_videos)):
            raise ValueError
        bad_apple_video_path = available_videos[bad_apple_choice_idx]

        # Step 2: Select the foreground video
        fg_choice_idx = int(input("Enter the number for your FOREGROUND video: ")) - 1
        if not (0 <= fg_choice_idx < len(available_videos)):
            raise ValueError
        fg_video_path = available_videos[fg_choice_idx]

        # Step 3: Select the background video
        bg_choice_idx = int(input("Enter the number for your BACKGROUND video: ")) - 1
        if not (0 <= bg_choice_idx < len(available_videos)):
            raise ValueError
        bg_video_path = available_videos[bg_choice_idx]
        
        # Check for unique selections
        if len({bad_apple_choice_idx, fg_choice_idx, bg_choice_idx}) != 3:
            print("Error: You must choose three different videos.")
            sys.exit(1)

    except (ValueError, IndexError):
        print("Invalid input or selection. Please enter a number from the list.")
        sys.exit(1)
        
    print(f"\nBad Apple video selected: {bad_apple_video_path}")
    print(f"Foreground video selected: {fg_video_path}")
    print(f"Background video selected: {bg_video_path}")
    
    # --- Adjust Thresholds for the first preview (at 17s) ---
    print("\n--- First Preview Session ---")
    final_fg_threshold, final_bg_threshold = adjust_thresholds(bad_apple_video_path, fg_video_path, bg_video_path, start_second=17, duration=10)

    # --- Adjust Thresholds for the second preview (at 1m 4s) ---
    print("\n--- Second Preview Session ---")
    final_fg_threshold, final_bg_threshold = adjust_thresholds(bad_apple_video_path, fg_video_path, bg_video_path, start_second=64, duration=10)
    
    proceed = input(f"\nAre you satisfied with the last preview? (y/n): ")
    if proceed.lower() != 'y':
        sys.exit(0)
    
    # --- Video Loading and Property Setup ---
    print("\nLoading videos and checking properties...")
    bad_apple_cap = cv2.VideoCapture(bad_apple_video_path)
    fg_cap = cv2.VideoCapture(fg_video_path)
    bg_cap = cv2.VideoCapture(bg_video_path)

    if not bad_apple_cap.isOpened() or not fg_cap.isOpened() or not bg_cap.isOpened():
        print("Error: Could not open one or more of the selected video files.")
        bad_apple_cap.release()
        fg_cap.release()
        bg_cap.release()
        sys.exit(1)

    # Get properties from the Bad Apple mask video
    width = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = bad_apple_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(bad_apple_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get properties from the user-selected videos
    fg_total_frames = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_total_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize the video writer for the output file
    temp_output_path = f"temp_output.mp4"
    output_video = cv2.VideoWriter(temp_output_path, FOURCC, fps, (width, height))

    # --- Frame Processing Loop ---
    print(f"\nProcessing video frames with foreground threshold {final_fg_threshold} and background threshold {final_bg_threshold}... This may take a while.")
    start_time = time.time() # Start the timer
    
    for current_frame_number in range(total_frames):
        # Read the current frame from the Bad Apple video
        ret_apple, apple_frame = bad_apple_cap.read()
        if not ret_apple:
            break

        # Calculate the corresponding frame number for the user's videos
        fg_frame_to_read = int(current_frame_number * (fg_total_frames / total_frames))
        bg_frame_to_read = int(current_frame_number * (bg_total_frames / total_frames))

        # Seek to the calculated frames to sync videos
        fg_cap.set(cv2.CAP_PROP_POS_FRAMES, fg_frame_to_read)
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_to_read)
        
        ret_fg, fg_frame = fg_cap.read()
        ret_bg, bg_frame = bg_cap.read()

        if not ret_fg or not ret_bg:
            # If a video ends, loop it from the beginning
            fg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_fg, fg_frame = fg_cap.read()
            ret_bg, bg_frame = bg_cap.read()

        # Resize user videos to match Bad Apple's dimensions
        fg_frame_resized = cv2.resize(fg_frame, (width, height))
        bg_frame_resized = cv2.resize(bg_frame, (width, height))
        
        # --- Luma Keying and Blending Logic ---
        gray_apple = cv2.cvtColor(apple_frame, cv2.COLOR_BGR2GRAY)
        
        _, fg_mask = cv2.threshold(gray_apple, final_fg_threshold, 255, cv2.THRESH_BINARY)
        _, bg_mask = cv2.threshold(gray_apple, final_bg_threshold, 255, cv2.THRESH_BINARY_INV)

        fg_masked = cv2.bitwise_and(fg_frame_resized, fg_frame_resized, mask=fg_mask)
        bg_masked = cv2.bitwise_and(bg_frame_resized, bg_frame_resized, mask=bg_mask)
        
        final_frame = cv2.add(fg_masked, bg_masked)
        
        # Write the final frame to the output video
        output_video.write(final_frame)
        
        # Print progress and ETA
        if (current_frame_number + 1) % 100 == 0 or current_frame_number + 1 == total_frames:
            elapsed_time = time.time() - start_time
            progress = (current_frame_number + 1) / total_frames
            
            if progress > 0:
                eta_seconds = (elapsed_time / progress) - elapsed_time
                minutes = int(eta_seconds // 60)
                seconds = int(eta_seconds % 60)
                sys.stdout.write(f"\rProgress: {progress*100:.2f}% | ETA: {minutes}m {seconds}s")
                sys.stdout.flush()

    # --- Clean up and release resources ---
    bad_apple_cap.release()
    fg_cap.release()
    bg_cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
    print("\n\nVideo frames processed. Now adding Bad Apple audio...")

    # --- Audio Mixing and Final Output ---
    try:
        # Load the audio directly from the Bad Apple video
        bad_apple_audio = VideoFileClip(bad_apple_video_path).audio
        
        # Load the temporary video and set the Bad Apple audio
        final_video_clip = VideoFileClip(temp_output_path)
        final_video_clip = final_video_clip.set_audio(bad_apple_audio)
        
        # Create the final output path and write the file
        output_video_path = f"output_{os.path.splitext(fg_video_path)[0]}_{os.path.splitext(bg_video_path)[0]}.mp4"
        final_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        
        # Clean up the temporary video file
        os.remove(temp_output_path)
        
        print(f"\nProcessing complete! 🎉 The final video with audio is saved as '{output_video_path}'")
        
    except Exception as e:
        print(f"Error during audio processing: {e}")
        print("A video file was created without audio. You may need to install ffmpeg for moviepy to work correctly.")

if __name__ == "__main__":
    main()
