import cv2
import numpy as np


def add_translucent_green_border(
    image: np.ndarray, thickness: int = 10, opacity: float = 0.5
) -> np.ndarray:
    """
    Adds a translucent green border to the edges of the input image.

    Parameters:
    - image: np.ndarray, the input image in OpenCV format (BGR).
    - thickness: int, the thickness of the border in pixels (default is 10).
    - opacity: float, the opacity of the border (0.0 to 1.0, default is 0.5).

    Returns:
    - np.ndarray, the image with a translucent green border.
    """
    # Ensure the opacity is within the valid range
    opacity = max(0.0, min(1.0, opacity))

    # Create a copy of the image to avoid modifying the original
    bordered_image = image.copy()

    # Define the green color in BGR format
    green_color = (0, 255, 0)

    # Create a border mask with the same shape as the image
    border_mask = np.zeros_like(bordered_image, dtype=np.uint8)

    # Draw a green rectangle on the border mask
    cv2.rectangle(
        border_mask,
        (0, 0),
        (bordered_image.shape[1], bordered_image.shape[0]),
        green_color,
        thickness,
    )

    # Blend the border mask with the original image using the specified opacity
    bordered_image = cv2.addWeighted(bordered_image, 1.0, border_mask, opacity, 0)

    return bordered_image


def tile_images(images: np.ndarray, max_width: int) -> np.ndarray:
    """Tiles a sequence of images into a single image in row-major order.

    Args:
        images (np.ndarray): A 4D numpy array of shape (num_images, height, width,
            channels) representing a sequence of images.
        max_width (int): The maximum number of images allowed in a single row.

    Returns:
        np.ndarray: A single image containing all the input images tiled together.
            Unfilled cells are filled with white pixels.
    """
    num_images, height, width, channels = images.shape

    # Calculate the number of rows and columns needed
    num_rows = (num_images + max_width - 1) // max_width
    num_cols = min(num_images, max_width)

    # Create a blank canvas to hold the tiled images
    canvas_height = num_rows * height
    canvas_width = num_cols * width
    canvas = (
        np.ones((canvas_height, canvas_width, channels), dtype=np.uint8) * 255
    )  # White background

    # Tile the images
    for i in range(num_images):
        row = i // max_width
        col = i % max_width
        y_start = row * height
        y_end = y_start + height
        x_start = col * width
        x_end = x_start + width
        canvas[y_start:y_end, x_start:x_end] = images[i]

    return canvas


def add_text_to_image(image: np.ndarray, text: str, top: bool = False) -> np.ndarray:
    """
    Adds text to the given image.

    Args:
        image (np.ndarray): Input image.
        text (str): Text to be added.
        top (bool, optional): Whether to add the text to the top or bottom of the image.

    Returns:
        np.ndarray: Image with text added.
    """
    width = image.shape[1]
    text_image = generate_text_image(width, text)
    if top:
        combined_image = np.vstack([text_image, image])
    else:
        combined_image = np.vstack([image, text_image])

    return combined_image


def generate_text_image(width: int, text: str) -> np.ndarray:
    """
    Generates an image of the given text with line breaks, honoring given width.

    Args:
        width (int): Width of the image.
        text (str): Text to be drawn.

    Returns:
        np.ndarray: Text drawn on white image with the given width.
    """
    # Define the parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_spacing = 10  # Spacing between lines in pixels

    # Calculate the maximum width and height of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    max_width = width - 20  # Allow some padding
    max_height = text_size[1] + line_spacing

    # Split the text into words
    words = text.split()

    # Initialize variables for text positioning
    x = 10
    y = text_size[1]

    to_draw = []

    # Iterate over the words and add them to the image
    num_rows = 1
    for word in words:
        # Get the size of the word
        word_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)

        # Check if adding the word exceeds the maximum width
        if x + word_size[0] > max_width:
            # Add a line break before the word
            y += max_height
            x = 10
            num_rows += 1

        # Draw the word on the image
        to_draw.append((word, x, y))

        # Update the position for the next word
        x += word_size[0] + 5  # Add some spacing between words

    # Create a blank white image with the calculated dimensions
    image = 255 * np.ones((max_height * num_rows, width, 3), dtype=np.uint8)
    for word, x, y in to_draw:
        cv2.putText(
            image,
            word,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return image


def combine_image_sequences(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
    """Combines two sequences of images horizontally into a single sequence.

    Takes two 4D numpy arrays representing image sequences and combines them side by
    side. If sequences have different lengths, the shorter one is padded by
    repeating its last frame.

    Args:
        seq1 (np.ndarray): First image sequence with shape (num_frames_1, height,
            width, 3).
        seq2 (np.ndarray): Second image sequence with shape (num_frames_2, height,
            width, 3).

    Returns:
        np.ndarray: Combined sequence with shape (max(num_frames_1, num_frames_2),
            height, width*2, 3).

    Raises:
        AssertionError: If the height, width, or number of channels differ between
            sequences.
    """
    # Assert that the last three dimensions of each input are equal
    assert (
        seq1.shape[1:] == seq2.shape[1:]
    ), "The last three dimensions of the input arrays must be equal."

    num_frames_1, H, W, C = seq1.shape
    num_frames_2 = seq2.shape[0]

    # Determine the maximum number of frames
    max_frames = max(num_frames_1, num_frames_2)

    # Pad the shorter sequence with its own last frame
    if num_frames_1 < max_frames:
        padding = np.tile(seq1[-1:], (max_frames - num_frames_1, 1, 1, 1))
        seq1 = np.concatenate([seq1, padding], axis=0)
    elif num_frames_2 < max_frames:
        padding = np.tile(seq2[-1:], (max_frames - num_frames_2, 1, 1, 1))
        seq2 = np.concatenate([seq2, padding], axis=0)

    # Combine the two sequences along the width dimension
    combined_seq = np.concatenate([seq1, seq2], axis=2)

    return combined_seq


def add_and_resize_4d(
    images_4d: np.ndarray, image_3d: np.ndarray, width: int
) -> np.ndarray:
    """Adds a 3D numpy array to the bottom of all frames in a 4D numpy array and resizes
    all frames to the specified width while preserving aspect ratio. Uses optimal
    interpolation methods: INTER_CUBIC for upscaling and INTER_AREA for downscaling.
    Ensures output dimensions are even numbers.

    Args:
        images_4d: 4D numpy array of shape (num_frames, height, width, channels)
        image_3d: 3D numpy array of shape (height, width, channels)
        width: Integer, the target width for all frames (will be made even if odd)

    Returns:
        4D numpy array of shape (num_frames, new_height, width, channels) with even
        dimensions
    """
    # Ensure the input arrays have the correct number of dimensions
    if images_4d.ndim != 4 or image_3d.ndim != 3:
        raise ValueError("Input arrays must be 4D and 3D respectively.")

    # Make sure target width is even
    width = width if width % 2 == 0 else width + 1

    num_frames, orig_height, orig_width, channels = images_4d.shape
    image_height, image_width, _ = image_3d.shape

    # Calculate new heights maintaining aspect ratio and ensure they're even
    new_frame_height = int(orig_height * (width / orig_width))
    new_frame_height = (
        new_frame_height if new_frame_height % 2 == 0 else new_frame_height + 1
    )

    new_image_height = int(image_height * (width / image_width))
    new_image_height = (
        new_image_height if new_image_height % 2 == 0 else new_image_height + 1
    )

    # Determine interpolation method for image_3d
    image_3d_interp = cv2.INTER_CUBIC if width > image_width else cv2.INTER_AREA

    # Resize the 3D image to match the target width while preserving aspect ratio
    resized_image_3d = cv2.resize(
        image_3d, (width, new_image_height), interpolation=image_3d_interp
    )

    # Determine interpolation method for frames
    frames_interp = cv2.INTER_CUBIC if width > orig_width else cv2.INTER_AREA

    # Resize frames one by one
    resized_frames = []
    for frame in images_4d:
        resized_frame = cv2.resize(
            frame, (width, new_frame_height), interpolation=frames_interp
        )
        resized_frames.append(resized_frame)

    # Stack back into 4D array
    resized_frames = np.stack(resized_frames)

    # Concatenate the resized 3D image to the bottom of each frame
    combined_frames = np.concatenate(
        (resized_frames, np.tile(resized_image_3d, (num_frames, 1, 1, 1))), axis=1
    )

    # Ensure final height is even
    final_height = combined_frames.shape[1]
    if final_height % 2 != 0:
        # Add a black row at the bottom
        padding = np.zeros(
            (num_frames, 1, width, channels), dtype=combined_frames.dtype
        )
        combined_frames = np.concatenate((combined_frames, padding), axis=1)

    return combined_frames


def images_to_video(video_frames: np.ndarray, output_path: str, fps: int = 30) -> None:
    """
    Convert a sequence of BGR frames to an MP4 video file.

    Args:
        video_frames (np.ndarray): Array of BGR frames with shape (frames, height, width, 3)
        output_path (str): Path where the output MP4 file will be saved
        fps (int, optional): Frames per second for the output video. Defaults to 30.

    Returns:
        None

    Raises:
        ValueError: If video_frames is not a 4D array with the correct shape
        ValueError: If the frames are not in BGR format (3 channels)
    """
    from moviepy.editor import ImageSequenceClip

    # Input validation
    if not isinstance(video_frames, np.ndarray) or len(video_frames.shape) != 4:
        raise ValueError(
            "video_frames must be a 4D numpy array with shape (frames, height, width, 3)"
        )

    if video_frames.shape[-1] != 3:
        raise ValueError("video_frames must have 3 channels (BGR format)")

    # Convert BGR to RGB for moviepy
    rgb_frames = []
    for frame in video_frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb_frame)

    # Create video clip
    clip = ImageSequenceClip(rgb_frames, fps=fps)

    # Write video file
    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        bitrate="10M",
        audio=False,
        verbose=False,
        threads=4,
    )

    # Clean up
    clip.close()

    print(f"Video saved to {output_path}")
