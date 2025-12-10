import numpy as np
import random
import scipy.ndimage as ndi
import numpy.linalg as npl
import torch

# ============================================================================
# Auxiliary Functions
# ============================================================================


def _calculate_all_permutations(item_list):
    """Calculate all permutations of items in nested lists.

    Args:
        item_list: List of lists to permute

    Returns:
        List of all permutations
    """
    if len(item_list) == 1:
        return [[i] for i in item_list[0]]
    sub_permutations = _calculate_all_permutations(item_list[1:])
    return [[i] + p for i in item_list[0] for p in sub_permutations]


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker with unique random seed.

    Args:
        worker_id: Worker process ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_weights_for_balanced_classes(labels: np.ndarray) -> list:
    """Create sampling weights for balanced classes.

    Args:
        labels: Array of class labels

    Returns:
        List of sampling weights for each sample
    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    return [n_samples / float(cnt_dict[label]) for label in labels]


def clip_and_scale(array: np.ndarray, max_hu: float = 400.0, min_hu: float = -1000.0) -> np.ndarray:
    """Clip and scale HU values to [0, 1] range.

    Args:
        array: Input array with HU values
        max_hu: Maximum HU value for clipping
        min_hu: Minimum HU value for clipping

    Returns:
        Normalized array in [0, 1] range
    """
    array = (array - min_hu) / (max_hu - min_hu)
    return np.clip(array, 0.0, 1.0)


def rotate_matrix_x(cos_angle: float, sin_angle: float) -> np.ndarray:
    """Create rotation matrix around X axis."""
    return np.array([[1, 0, 0], [0, cos_angle, -sin_angle], [0, sin_angle, cos_angle]])


def rotate_matrix_y(cos_angle: float, sin_angle: float) -> np.ndarray:
    """Create rotation matrix around Y axis."""
    return np.array([[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]])


def rotate_matrix_z(cos_angle: float, sin_angle: float) -> np.ndarray:
    """Create rotation matrix around Z axis."""
    return np.array([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]])


def sample_random_coordinate_on_sphere(radius: float) -> np.ndarray:
    """Sample a random 3D coordinate on sphere surface.

    Args:
        radius: Sphere radius

    Returns:
        3D coordinate on sphere surface
    """
    random_nums = np.random.normal(size=(3,))
    if np.all(random_nums == 0):
        random_nums = np.array([1.0, 0.0, 0.0])
    return random_nums / np.linalg.norm(random_nums) * radius


def volume_transform(
    image: np.ndarray,
    voxel_spacing: np.ndarray,
    transform_matrix: np.ndarray,
    center: Optional[np.ndarray] = None,
    output_shape: Optional[np.ndarray] = None,
    output_voxel_spacing: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Transform 3D volume using affine transformation.

    Args:
        image: Input 3D image array
        voxel_spacing: Voxel spacing for each dimension
        transform_matrix: Affine transformation matrix
        center: Center point for transformation
        output_shape: Shape of output image
        output_voxel_spacing: Voxel spacing for output
        **kwargs: Additional arguments for scipy.ndimage.affine_transform

    Returns:
        Transformed 3D image
    """
    if "offset" in kwargs or "output_shape" in kwargs:
        raise ValueError("Cannot supply 'offset' or 'output_shape' - already used by this function")

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must match image dimensions")

    # Calculate voxel center
    if center is None:
        voxel_center = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError("Center must match image dimensions")
        voxel_center = np.asarray(center) / voxel_spacing

    # Set output voxel spacing
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = voxel_spacing
        else:
            output_voxel_spacing = np.ones(image.ndim)
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    transform_matrix = np.asarray(transform_matrix)
    if transform_matrix.shape[1] != image.ndim or transform_matrix.shape[0] != image.ndim:
        raise ValueError("Transform matrix must be square and match image dimensions")

    # Normalize the transform matrix
    transform_matrix = (transform_matrix.T / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))).T
    transform_matrix = np.linalg.inv(transform_matrix.T)

    # Forward matrix: input image space -> result image space
    forward_matrix = np.dot(np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix), np.diag(voxel_spacing))

    # Calculate output shape if not provided
    if output_shape is None:
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxel_center, image.shape)]
        image_corners = _calculate_all_permutations(image_axes)
        transformed_corners = [np.dot(forward_matrix, corner) for corner in image_corners]

        output_shape = [
            1
            + int(
                np.ceil(
                    2 * max(abs(np.min(transformed_corners, axis=0)[i]), abs(np.max(transformed_corners, axis=0)[i]))
                )
            )
            for i in range(len(voxel_center))
        ]
    else:
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError("Output shape must match transform matrix dimensions")

    output_shape = np.array(output_shape)

    # Calculate backwards matrix for slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxel_center - backwards_matrix.dot((output_shape - 1) / 2.0)

    return ndi.affine_transform(
        image, backwards_matrix, offset=target_image_offset, output_shape=output_shape, **kwargs
    )


def extract_patch(
    ct_data: np.ndarray,
    coord: np.ndarray,
    src_voxel_origin: np.ndarray,
    src_world_matrix: np.ndarray,
    src_voxel_spacing: np.ndarray,
    output_shape: Tuple[int, int, int] = (64, 64, 64),
    voxel_spacing: Tuple[float, float, float] = (50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations: Optional[Tuple] = None,
    translations: Optional[bool] = None,
    coord_space_world: bool = False,
    mode: str = "2D",
    model_name: str = "",
) -> np.ndarray:
    """Extract a patch from CT volume with optional augmentations.

    Args:
        ct_data: CT volume data
        coord: Coordinate for patch center
        src_voxel_origin: Origin of voxel space
        src_world_matrix: World to voxel transformation matrix
        src_voxel_spacing: Voxel spacing
        output_shape: Shape of output patch
        voxel_spacing: Voxel spacing for output
        rotations: Rotation ranges for augmentation (tuple of ranges)
        translations: Whether to apply random translations
        coord_space_world: Whether coordinates are in world space
        mode: "2D" or "3D"
        model_name: Name of the model for channel determination

    Returns:
        Extracted and normalized patch
    """
    transform_matrix = np.eye(3)

    # Apply rotations
    if rotations is not None:
        angle_x = np.random.uniform(*rotations[0])
        angle_y = np.random.uniform(*rotations[1])
        angle_z = np.random.uniform(*rotations[2])

        rot_x = rotate_matrix_x(np.cos(angle_x), np.sin(angle_x))
        rot_y = rotate_matrix_y(np.cos(angle_y), np.sin(angle_y))
        rot_z = rotate_matrix_z(np.cos(angle_z), np.sin(angle_z))

        transform_matrix = rot_z @ rot_y @ rot_x

    # Apply translations
    if translations is not None:
        translation_radius = np.random.uniform(0, min(output_shape) / 4)
        translation = sample_random_coordinate_on_sphere(translation_radius)
        coord = coord + translation

    # Normalize transform matrix
    transform_matrix = (transform_matrix.T / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))).T

    inv_src_matrix = np.linalg.inv(src_world_matrix)

    # Calculate override coordinates
    if coord_space_world:
        override_coord = coord
    else:
        override_coord = src_voxel_origin + coord * src_voxel_spacing

    override_matrix = (inv_src_matrix.dot(transform_matrix.T) * src_voxel_spacing).T

    patch = volume_transform(
        ct_data,
        src_voxel_spacing,
        override_matrix,
        center=override_coord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )

    # Process patch based on mode
    if mode == "2D":
        mid_slice = patch.shape[0] // 2
        patch = patch[mid_slice, :, :]
        patch = clip_and_scale(patch)

        # Determine if model needs RGB channels
        imagenet_models = [
            "unet-2d",
            "fpn-2d",
            "pspnet-2d",
            "deeplabv3-2d",
            "upernet-2d",
            "dpt-2d",
            "linknet-2d",
            "manet-2d",
        ]
        is_nnunet = model_name.startswith("nnunet")
        needs_rgb = model_name in imagenet_models and not is_nnunet

        if needs_rgb:
            # Convert to RGB by repeating channel
            patch = np.stack([patch, patch, patch], axis=0)
        else:
            # Keep single channel
            patch = patch[np.newaxis, :, :]
    else:  # 3D
        patch = clip_and_scale(patch)
        patch = patch[np.newaxis, :, :, :]

    return patch.astype(np.float32)
