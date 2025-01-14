#include <iostream>
#include <string>

// Interface class
class Shape {
public:
    // Pure virtual method to calculate area
    [[nodiscard]] virtual double calculateArea() const = 0;

    // Pure virtual method to calculate perimeter
    [[nodiscard]] virtual double calculatePerimeter() const = 0;

    // Pure virtual method to display details
    virtual void display() const = 0;

    // Virtual destructor to allow proper cleanup of derived class objects
    virtual ~Shape() = default;

    // Explicitly delete copy and move constructors/assignment
    Shape(const Shape&) = delete;
    Shape& operator=(const Shape&) = delete;
    Shape(Shape&&) = delete;
    Shape& operator=(Shape&&) = delete;

    Shape() = default;
};

// Example derived class: Circle
class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    // Implement calculateArea
    [[nodiscard]] double calculateArea() const override { return 3.14159 * radius * radius; }

    // Implement calculatePerimeter
    [[nodiscard]] double calculatePerimeter() const override { return 2 * 3.14159 * radius; }

    // Implement display
    void display() const override {
        std::cout << "Circle: Radius = " << radius << ", Area = " << calculateArea()
                  << ", Perimeter = " << calculatePerimeter() << '\n';
    }
};

// Example derived class: Rectangle
class Rectangle : public Shape {
private:
    double length, width;

public:
    Rectangle(double l, double w) : length(l), width(w) {}

    // Implement calculateArea
    [[nodiscard]] double calculateArea() const override { return length * width; }

    // Implement calculatePerimeter
    [[nodiscard]] double calculatePerimeter() const override { return 2 * (length + width); }

    // Implement display
    void display() const override {
        std::cout << "Rectangle: Length = " << length << ", Width = " << width
                  << ", Area = " << calculateArea() << ", Perimeter = " << calculatePerimeter()
                  << '\n';
    }
};
