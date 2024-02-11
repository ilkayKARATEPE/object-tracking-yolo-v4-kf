import cv2 as cv
import os
import time
from typing import List, Tuple, Dict, Optional, Set, Union
from models.point import Point
from utils import calculate_center_point, euclidean_distance
from core.kalman_filter import KalmanFilter
from .object_detection import ObjectDetectionYoloV4


class ObjectTracking:
    def __init__(
        self,
        detector: Union[ObjectDetectionYoloV4],
        path_to_video: str,
        detect_time: Optional[float] = None,
        kf_time: Optional[float] = None,
    ):
        self._path_to_video = path_to_video
        self._od: Union[ObjectDetectionYoloV4] = detector
        self._capture: cv.VideoCapture = cv.VideoCapture(self._path_to_video)
        self._center_pts_prev_frame: List[Point] = []
        self._tracking_objects: Dict[int, Point] = {}
        self._tracking_id: int = 0
        self._frame_count: int = 0
        self._kf: KalmanFilter = KalmanFilter()
        self._kf_prev_prediction: Point = None
        self._detect_time: Optional[float] = detect_time
        self._kf_time: Optional[float] = kf_time

        # Using XOR to ensure either both are defined or both are None
        if bool(self._detect_time) ^ bool(self._kf_time):
            raise ValueError(
                "Both _detect_time and _kf_time must be defined or both must be None."
            )

        self._interval_mode = (
            self._detect_time is not None and self._kf_time is not None
        )

    def track(self):
        start = time.time()
        detect_mode = True
        while True:
            has_frame, frame = self._capture.read()
            if not has_frame:
                break
            self._frame_count += 1

            bounding_boxes = (
                [] if not detect_mode else self._construct_bounding_boxes(frame)
            )
            center_pts_cur_frame = (
                [] if not detect_mode else self._find_center_points(bounding_boxes)
            )

            if len(center_pts_cur_frame) == 1:
                self._kf_prev_prediction = self._kf.predict(*center_pts_cur_frame[0])
                cv.circle(
                    frame,
                    center=self._kf_prev_prediction,
                    radius=20,
                    color=(255, 0, 0),
                    thickness=4,
                )

            elif (
                self._kf_prev_prediction is not None and len(center_pts_cur_frame) == 0
            ):
                self._kf_prev_prediction = self._kf.predict(*self._kf_prev_prediction)
                cv.circle(
                    frame,
                    center=self._kf_prev_prediction,
                    radius=20,
                    color=(255, 0, 0),
                    thickness=4,
                )

            self._draw_center_points(frame, center_pts_cur_frame)
            self._draw_bounding_boxes(frame, bounding_boxes)

            # Compare with previous frame's center points and update tracking
            self._compare_center_points(center_pts_cur_frame)

            self._draw_labels(frame)

            self._center_pts_prev_frame = center_pts_cur_frame.copy()

            cv.imshow(self._path_to_video, frame)
            key: int = cv.waitKey(100)
            if key == 27:  # Escape key
                break

            # Switch from detect mode to kf mode
            if self._interval_mode:
                end = time.time()

                if (detect_mode and (end - start >= self._detect_time)) or (
                    not detect_mode and (end - start >= self._kf_time)
                ):
                    print(f"Switch from {detect_mode=} to {not detect_mode}")
                    detect_mode = not detect_mode
                    start = end

        self._clean_up()

    def _construct_bounding_boxes(self, cur_frame) -> List[Tuple[Point, Point]]:
        start = time.time()
        (class_ids, scores, boxes) = self._od.detect(frame=cur_frame)
        print(f"{round(time.time() - start, 2)}s for detection time")
        bounding_boxes: List[Tuple[Point, Point]] = []
        for box in boxes:
            (x, y, width, height) = box
            pt1 = Point(x, y)
            pt2 = Point(x + width, y + height)
            bounding_boxes.append((pt1, pt2))

        return bounding_boxes

    def _find_center_points(
        self, bounding_boxes: List[Tuple[Point, Point]]
    ) -> List[Point]:
        return [calculate_center_point(pt1, pt2) for pt1, pt2 in bounding_boxes]

    def _compare_center_points(self, cur_ctr_pts: List[Point]) -> None:
        """
        Compares current center points with the tracked objects. In the first two frames,
        it attempts to match the current center points with the previous frame's center points.
        For subsequent frames, it matches current center points with existing tracked objects.

        If a match is found with a tracked object, the object's position is updated.
        If no match is found, the point is considered a new object and assigned a new tracking ID.

        This method combines matching against tracked objects and previous frame center points
        to improve tracking accuracy, especially for objects that may not be detected in the initial frames.

        Args:
            cur_ctr_pts (List[Point]): A list of current frame's center points derived from detected objects.
        """
        if self._frame_count <= 2:
            # In the first two frames, match with previous frame's center points
            for cur_ctr_pt in cur_ctr_pts:
                tracking_id, has_match = self._match_prev_frame_ctr_pts(
                    cur_pt=cur_ctr_pt
                )
                if has_match:
                    self._update_tracked_object(
                        tracking_id=tracking_id, new_ctr_pt=cur_ctr_pt, add=True
                    )
                    self._tracking_id += 1
        else:
            # Prepare a list to track IDs of objects to remove
            to_remove = set()

            # In order to determine which center points should be added as new 'detected' objects to keep track off.
            matched_ctr_pts = set()

            for tracking_id, tracked_ctr_pt in self._tracking_objects.items():
                tracking_id, cur_ctr_pt, has_match = self._match_tracking_pts(
                    cur_ctr_pts=cur_ctr_pts,
                    cur_tracked_ctr_pt=tracked_ctr_pt,
                    cur_tracked_ctr_pt_tracking_id=tracking_id,
                )
                if not has_match:
                    # Mark the point for removal since we have lost it
                    to_remove.add(tracking_id)
                    continue

                # Since current center point is already associated with some tracked center point and we
                # don't want to re-associate it.
                cur_ctr_pts.remove(cur_ctr_pt)

                # Update tracked object with new position that was found
                self._update_tracked_object(
                    tracking_id=tracking_id, new_ctr_pt=cur_ctr_pt
                )
                matched_ctr_pts.add(cur_ctr_pt)

            # Remove tracked objects that were not matched with current center points
            self._remove_tracked_objects(tracking_ids=to_remove)

            # Add current center points that were not matched with tracked objects
            for ctr_pt_cur_frame in cur_ctr_pts:
                if ctr_pt_cur_frame not in matched_ctr_pts:
                    self._update_tracked_object(
                        tracking_id=self._tracking_id,
                        new_ctr_pt=ctr_pt_cur_frame,
                        add=True,
                    )
                    self._tracking_id += 1

    def _match_prev_frame_ctr_pts(
        self, cur_pt: Point, threshold: int = 20
    ) -> Tuple[int, bool]:
        """
        Matches the current point with the previous frame's center points within a threshold.
        Returns the matching tracking ID and a boolean indicating if a match was found.
        """
        for prev_pt in self._center_pts_prev_frame:
            if euclidean_distance(cur_pt, prev_pt) < threshold:
                return (self._tracking_id, True)
        return (self._tracking_id, False)

    def _match_tracking_pts(
        self,
        cur_ctr_pts: List[Point],
        cur_tracked_ctr_pt: Point,
        cur_tracked_ctr_pt_tracking_id: int,
        threshold: int = 20,
    ) -> Tuple[int, Optional[Point], bool]:
        """
        Matches a given tracked point with the current frame's center points. It checks if any of the
        current center points are within a specified distance threshold from the tracked point.

        If a matching point is found, it updates the tracked object's position with this new point
        and returns the tracking ID and a `True` flag. If no matching point is found, it returns
        the original tracking ID and a `False` flag.

        Args:
            cur_ctr_pts (List[Point]): List of center points in the current frame.
            cur_tracked_ctr_pt (Point): The center point of the tracked object to be matched.
            cur_tracked_ctr_pt_tracking_id (int): The tracking ID of the current tracked object.
            threshold (int, optional): The distance threshold for considering a match. Defaults to 20.

        Returns:
            Tuple[int, Optional[Point], bool]: A tuple containing the tracking ID, the Point of current frame center points (optional) and a boolean indicating if a match was found.
        """

        for cur_ctr_pt in cur_ctr_pts:
            if euclidean_distance(cur_tracked_ctr_pt, cur_ctr_pt) < threshold:
                self._update_tracked_object(
                    tracking_id=cur_tracked_ctr_pt_tracking_id,
                    new_ctr_pt=cur_tracked_ctr_pt,
                )
                return (cur_tracked_ctr_pt_tracking_id, cur_ctr_pt, True)
        return (cur_tracked_ctr_pt_tracking_id, None, False)

    def _update_tracked_object(
        self, tracking_id: int, new_ctr_pt: Point, add: bool = False
    ) -> None:
        """
        Updates the position of a tracked object in the tracking system. If the specified tracking ID
        exists in the list of currently tracked objects, this method updates the corresponding center
        point with the new value provided.

        It's crucial that this method is called only with valid tracking IDs. If a tracking ID
        that does not exist in the current list of tracked objects is passed, the method raises
        a KeyError, indicating an attempt to update a non-existent tracking entity.

        Args:
            tracking_id (int): The ID of the tracked object to be updated.
            new_ctr_pt (Point): The new center point for the tracked object.
            add (bool): Whether to add or just update an item.

        Raises:
            KeyError: If `tracking_id` is not found in the current list of tracked objects.
        """
        if not add and tracking_id not in self._tracking_objects:
            raise KeyError(f"{tracking_id} not in _tracking_objects.")
        self._tracking_objects[tracking_id] = new_ctr_pt

    def _remove_tracked_objects(self, tracking_ids: Set[int]) -> None:
        """
        Removes a list of tracked objects from the tracking system. This method iterates over
        the provided list of tracking IDs and removes each corresponding object from the
        internal tracking objects dictionary.

        It is important to ensure that all provided tracking IDs exist in the current tracking
        system. If an ID is not found, the method raises a KeyError, signifying an attempt to
        remove a non-existent tracking entity.

        This method is typically used to remove objects that are no longer in the field of view
        or when they are no longer relevant for tracking.

        Args:
            tracking_ids (Set[int]): A set of tracking IDs of the objects to be removed.

        Raises:
            KeyError: If any of the `tracking_ids` is not found in the current list of tracked objects.
        """
        for tracking_id in tracking_ids:
            if tracking_id not in self._tracking_objects:
                raise KeyError(f"{tracking_id} not in _tracking_objects.")
            self._tracking_objects.pop(tracking_id)

    def _draw_center_points(self, frame, center_points: List[Point]) -> None:
        if center_points is None:
            raise ValueError("Center Points cannot be None.")
        for cp in center_points:
            cv.circle(
                img=frame,
                center=(cp.x, cp.y),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )

    def _draw_bounding_boxes(
        self, frame, bounding_boxes: List[Tuple[Point, Point]]
    ) -> None:
        for pt1, pt2 in bounding_boxes:
            cv.rectangle(
                img=frame,
                pt1=(pt1.x, pt1.y),
                pt2=(pt2.x, pt2.y),
                color=(0, 255, 0),
                thickness=2,
            )

    def _draw_labels(self, frame) -> None:
        for tracking_id, ctr_pt in self._tracking_objects.items():
            cv.putText(
                img=frame,
                text=str(tracking_id),
                org=(ctr_pt.x, ctr_pt.y - 7),
                fontFace=0,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
            )

    def _clean_up(self) -> None:
        self._capture.release()
        cv.destroyAllWindows()
