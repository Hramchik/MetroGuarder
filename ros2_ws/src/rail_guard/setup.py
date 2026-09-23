from setuptools import setup

package_name = "rail_guard"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, f"{package_name}.lib", f"{package_name}.nodes"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hram",
    maintainer_email="sanek.falaleev@gmail.com",
    description="Мониторинг габарита беспилотного поезда метро по данным 3D-лидара",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dataset_player = rail_guard.nodes.dataset_player_node:main",
            "obstacle_detector = rail_guard.nodes.obstacle_detector_node:main",
            "result_monitor = rail_guard.nodes.result_monitor_node:main",
        ],
    },
)
