"""
base_env.py
-----------
Lớp trừu tượng (Abstract Base Class) cho tất cả môi trường RL trong dự án.
Mọi môi trường cụ thể phải kế thừa lớp này và cài đặt đầy đủ các phương thức.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseEnv(ABC):
    """
    Lớp cơ sở cho môi trường RL tự viết.
    Chuẩn thiết kế tương tự Gymnasium nhưng KHÔNG dùng Gymnasium.
    """

    # ------------------------------------------------------------------ #
    #  Phương thức bắt buộc                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        """
        Khởi tạo lại môi trường về trạng thái đầu.

        Parameters
        ----------
        seed : int, optional
            Hạt ngẫu nhiên để tái lập kết quả.

        Returns
        -------
        observation : int hoặc array
            Trạng thái ban đầu sau khi mã hóa.
        info : dict
            Thông tin bổ sung (có thể rỗng).
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Thực hiện một bước hành động.

        Parameters
        ----------
        action : int
            Hành động agent chọn.

        Returns
        -------
        observation : int hoặc array
            Trạng thái kế tiếp sau khi mã hóa.
        reward : float
            Phần thưởng tức thời.
        terminated : bool
            True nếu episode kết thúc tự nhiên (đến đích, va chạm).
        truncated : bool
            True nếu episode bị cắt ngắn (hết max_steps).
        info : dict
            Thông tin bổ sung: collision, reached_goal, ...
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> str:
        """
        Trả về chuỗi mô tả trạng thái hiện tại dưới dạng văn bản.

        Returns
        -------
        str
            Biểu diễn grid dạng văn bản có thể in ra terminal.
        """
        raise NotImplementedError

    @abstractmethod
    def state_encoder(self, state: tuple) -> int:
        """
        Mã hóa tuple (r, c, heading, goal_id) → chỉ số nguyên.

        Parameters
        ----------
        state : tuple
            Trạng thái thô (row, col, heading, goal_id).

        Returns
        -------
        int
            Chỉ số trạng thái trong [0, n_states).
        """
        raise NotImplementedError

    @abstractmethod
    def state_decoder(self, encoded: int) -> tuple:
        """
        Giải mã chỉ số nguyên → tuple (r, c, heading, goal_id).

        Parameters
        ----------
        encoded : int
            Chỉ số trạng thái.

        Returns
        -------
        tuple
            (row, col, heading, goal_id)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Thuộc tính bắt buộc (property)                                      #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Tổng số trạng thái trong không gian trạng thái."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Tổng số hành động hợp lệ."""
        raise NotImplementedError
