from unittest.mock import patch

from worker import main


def test_worker_starts():
    with patch("worker.Worker") as worker_cls:
        main()

    worker_cls.return_value.work.assert_called_once()
